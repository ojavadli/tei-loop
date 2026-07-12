"""Discrete Adaptive Cubic Regularization (D-ARC) for prompt optimization.

D-ARC is an ARC-inspired, candidate-restricted local-surrogate method for
DISCRETE prompt optimization. It adapts the *control structure* of adaptive
cubic regularisation — a fitted local quadratic model plus a cubic distance
penalty, step acceptance by predicted-vs-actual reduction ratio rho, and an
adaptive regularisation weight sigma — to a setting where the decision
variable is prompt TEXT, not a vector.

IMPORTANT SCOPE / HONESTY NOTES (do not weaken these in docs or papers):
  * Prompt text is discrete: there is no exact gradient or Hessian of an LLM
    with respect to its prompt. The g_k and B_k used here are FITTED SURROGATE
    COEFFICIENTS in a deterministic prompt-feature space (see
    ``PromptFeatureMap``), estimated by weighted ridge regression from
    previously evaluated prompts. They are NOT true derivatives of the LLM.
  * Consequently D-ARC does NOT inherit the convergence theory or the
    O(epsilon^-3/2) worst-case complexity guarantees of classical continuous
    ARC. Those results assume exact (or controlled-inexact) derivatives of a
    smooth objective; nothing of the sort holds here.
  * The "subproblem" min_s m_k(s) is solved only over the FINITE candidate set
    returned by an LLM proposal call (candidate-restricted approximation),
    never over the continuous feature space — we never pretend to decode an
    arbitrary feature vector back into text.

Mathematical foundations (for the continuous method this is inspired by):
  * Nesterov & Polyak (2006), "Cubic regularization of Newton method and its
    global performance", Math. Programming 108:177-205.
    https://doi.org/10.1007/s10107-006-0706-8
  * Cartis, Gould & Toint, "Adaptive cubic regularisation methods for
    unconstrained optimization" (ARC part I).
    https://people.maths.ox.ac.uk/cartis/papers/ARCpI.pdf

Scalarization: for prompt p with approved metrics j scored on 0-100,
    z_j(p) = clamp(score_j(p)/100, 0, 1)
    U(p)   = sum_j w_j z_j(p)          (weights normalized, nonnegative)
    F(p)   = 1 - U(p)
D-ARC minimizes F; both U and F are stored on every record.

Cubic model at incumbent p_k with x_k = phi(p_k), proposal p with
s = phi(p) - phi(p_k), z = Q_k^T s (Q_k = fitted local subspace):
    m_k(p) = F(p_k) + g_k^T z + 0.5 z^T B_k z + (sigma_k/3) * ||s||_2^3
The cubic penalty uses the FULL feature-space norm ||s||, not the projected
norm, so novel directions outside the fitted subspace are penalized.

This module is pure computation + state: no LLM calls, no file IO, no
asyncio. Callers supply proposals and evaluation results; the controller
supplies selection, acceptance, sigma adaptation, and audit records.
"""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional

try:  # numpy is required only for cubic/hybrid modes, not for pareto
    import numpy as _np
except ImportError:  # pragma: no cover - exercised via _require_numpy test
    _np = None


def _require_numpy():
    if _np is None:
        raise ImportError(
            "D-ARC (optimizer modes 'cubic' and 'hybrid') requires numpy. "
            "Install it with:  pip install 'tei-loop[cubic]'  or  pip install numpy. "
            "The default 'pareto' mode does not need numpy."
        )
    return _np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CubicConfig:
    """Validated D-ARC configuration (defaults are the pre-registered values)."""
    sigma0: float = 1.0
    sigma_min: float = 1e-4
    sigma_max: float = 1e4
    eta1: float = 0.10          # successful threshold
    eta2: float = 0.90          # very-successful threshold
    gamma_dec: float = 0.50     # sigma multiplier on very successful
    gamma_inc: float = 2.00     # sigma multiplier on failure
    warmup_evals: int = 5
    proposals_per_iteration: int = 4
    window: int = 20            # max nearest observations for the surrogate
    max_rank: int = 4
    ridge_lambda: float = 1e-3
    eig_clip: float = 10.0
    pred_floor: float = 1e-8
    patience: int = 3           # consecutive failed/no-descent iterations before stop
    hash_dims: int = 256
    ngram_sizes: tuple = (3, 4, 5)
    utility_improve_eps: float = 1e-12   # strict warm-up improvement margin
    dedup_feature_eps: float = 1e-9      # feature-distance dedup threshold

    def __post_init__(self):
        if not (0.0 < self.sigma_min <= self.sigma0 <= self.sigma_max):
            raise ValueError("require 0 < sigma_min <= sigma0 <= sigma_max")
        if not (0.0 < self.eta1 < self.eta2 < 1.0):
            raise ValueError("require 0 < eta1 < eta2 < 1")
        if not (0.0 < self.gamma_dec < 1.0 < self.gamma_inc):
            raise ValueError("require 0 < gamma_dec < 1 < gamma_inc")
        if self.warmup_evals < 0 or self.proposals_per_iteration < 1:
            raise ValueError("warmup_evals >= 0 and proposals_per_iteration >= 1")
        if self.window < 3 or self.max_rank < 1:
            raise ValueError("window >= 3 and max_rank >= 1")
        if self.ridge_lambda <= 0 or self.eig_clip <= 0 or self.pred_floor <= 0:
            raise ValueError("ridge_lambda, eig_clip, pred_floor must be > 0")
        if self.patience < 1:
            raise ValueError("patience >= 1")

    def to_dict(self) -> dict:
        return {
            "sigma0": self.sigma0, "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max, "eta1": self.eta1, "eta2": self.eta2,
            "gamma_dec": self.gamma_dec, "gamma_inc": self.gamma_inc,
            "warmup_evals": self.warmup_evals,
            "proposals_per_iteration": self.proposals_per_iteration,
            "window": self.window, "max_rank": self.max_rank,
            "ridge_lambda": self.ridge_lambda, "eig_clip": self.eig_clip,
            "pred_floor": self.pred_floor, "patience": self.patience,
            "hash_dims": self.hash_dims, "ngram_sizes": list(self.ngram_sizes),
        }


def scalar_utility(metric_scores_0_100: dict[str, float],
                   weights: dict[str, float]) -> float:
    """U(p) = sum_j w_j * clamp(score_j/100, 0, 1) with normalized weights."""
    if not weights:
        return 0.0
    wsum = sum(max(0.0, w) for w in weights.values())
    if wsum <= 0:
        return 0.0
    u = 0.0
    for name, w in weights.items():
        z = min(1.0, max(0.0, metric_scores_0_100.get(name, 0.0) / 100.0))
        u += max(0.0, w) / wsum * z
    return u


def edit_radius(sigma: float) -> float:
    """Target relative edit radius handed to the proposal generator."""
    return min(1.0, max(0.05, 1.0 / math.sqrt(max(sigma, 1e-300))))


# ---------------------------------------------------------------------------
# Deterministic prompt feature map
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+")


class PromptFeatureMap:
    """Canonical deterministic prompt -> R^(hash_dims+8) feature map.

    Text normalization (in this exact order):
      1. Unicode NFKC.
      2. Uppercase ratio is computed on the NFKC text BEFORE lowercasing
         (the original casing must survive long enough for this feature);
         line-based structural counts also use the NFKC (pre-collapse) text.
      3. Lowercase.
      4. Collapse consecutive whitespace to one space.
      5. Strip leading/trailing whitespace.

    Hashed features: signed character n-grams (n = 3,4,5) over the normalized
    string wrapped in explicit boundary markers '^'...'$'. Bucket and sign are
    both derived from hashlib.blake2b of the n-gram (never Python hash(), so
    the map is identical across processes and PYTHONHASHSEED values).

    Structural features (fixed, documented scaling):
      1. log1p(character_count) / 10
      2. log1p(word_count) / 8
      3. log1p(nonempty_line_count) / 5
      4. log1p(markdown_header_count) / 3
      5. log1p(bullet_count) / 5
      6. log1p(digit_count) / 5
      7. unique_word_count / max(word_count, 1)
      8. uppercase_character_count / max(alpha_character_count, 1)

    The complete vector is L2-normalized. Vectors are cached by the SHA-256
    of the normalized prompt text.
    """

    def __init__(self, hash_dims: int = 256, ngram_sizes: tuple = (3, 4, 5)):
        _require_numpy()
        self.hash_dims = int(hash_dims)
        self.ngram_sizes = tuple(int(n) for n in ngram_sizes)
        self._cache: dict[str, Any] = {}

    # -- normalization ----------------------------------------------------
    @staticmethod
    def normalize_text(text: str) -> str:
        nfkc = unicodedata.normalize("NFKC", text or "")
        lowered = nfkc.lower()
        return re.sub(r"\s+", " ", lowered).strip()

    def prompt_key(self, text: str) -> str:
        """SHA-256 of the normalized prompt — the candidate IDENTITY key used
        for deduplication and observation bookkeeping (case- and
        whitespace-insensitive by construction of the normalization)."""
        return hashlib.sha256(self.normalize_text(text).encode("utf-8")).hexdigest()

    @staticmethod
    def _cache_key(text: str) -> str:
        """Internal vector-cache key: SHA-256 of the NFKC text BEFORE
        lowercasing. The uppercase-ratio structural feature depends on the
        original casing, so caching by the lowercased identity key would
        return a wrong vector for case-variant prompts."""
        nfkc = unicodedata.normalize("NFKC", text or "")
        return hashlib.sha256(nfkc.encode("utf-8")).hexdigest()

    # -- features ----------------------------------------------------------
    def _structural(self, nfkc: str) -> list[float]:
        char_count = len(nfkc)
        words = nfkc.split()
        word_count = len(words)
        lines = nfkc.splitlines()
        nonempty_lines = sum(1 for ln in lines if ln.strip())
        headers = sum(1 for ln in lines if _HEADER_RE.match(ln))
        bullets = sum(1 for ln in lines if _BULLET_RE.match(ln))
        digit_count = sum(1 for c in nfkc if c.isdigit())
        unique_words = len({w.lower() for w in words})
        alpha_chars = sum(1 for c in nfkc if c.isalpha())
        upper_chars = sum(1 for c in nfkc if c.isupper())
        return [
            math.log1p(char_count) / 10.0,
            math.log1p(word_count) / 8.0,
            math.log1p(nonempty_lines) / 5.0,
            math.log1p(headers) / 3.0,
            math.log1p(bullets) / 5.0,
            math.log1p(digit_count) / 5.0,
            unique_words / max(word_count, 1),
            upper_chars / max(alpha_chars, 1),
        ]

    def features(self, text: str):
        np = _require_numpy()
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]

        nfkc = unicodedata.normalize("NFKC", text or "")
        norm = self.normalize_text(text)
        vec = np.zeros(self.hash_dims + 8, dtype=np.float64)

        marked = f"^{norm}$"
        for n in self.ngram_sizes:
            if len(marked) < n:
                continue
            for i in range(len(marked) - n + 1):
                gram = marked[i:i + n]
                digest = hashlib.blake2b(
                    f"{n}|{gram}".encode("utf-8"), digest_size=8
                ).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.hash_dims
                sign = 1.0 if (digest[4] & 1) else -1.0
                vec[bucket] += sign

        vec[self.hash_dims:] = self._structural(nfkc)

        n2 = float(np.linalg.norm(vec))
        if n2 > 0:
            vec = vec / n2
        self._cache[key] = vec
        return vec


# ---------------------------------------------------------------------------
# Local quadratic surrogate (fitted; NOT derivatives)
# ---------------------------------------------------------------------------

@dataclass
class SurrogateFit:
    """A fitted local surrogate around the incumbent.

    kind: "quadratic" (g + B in an r-dim subspace), "linear" (B = 0), or
    "none" (insufficient observations; warm-up).
    All diagnostics required for auditability are recorded here.
    """
    kind: str = "none"
    rank: int = 0
    n_observations: int = 0
    g: Any = None                     # (r,) ndarray or None
    B: Any = None                     # (r,r) symmetric ndarray or None
    Q: Any = None                     # (d,r) ndarray or None
    weighted_rmse: Optional[float] = None
    ridge_lambda: float = 1e-3
    condition_number: Optional[float] = None
    eig_range_unclipped: Optional[list] = None
    eig_range_clipped: Optional[list] = None
    locality_bandwidth: Optional[float] = None
    reason: str = ""                  # why kind == "none"/"linear", if so

    def diagnostics(self) -> dict:
        return {
            "kind": self.kind,
            "rank": self.rank,
            "n_observations": self.n_observations,
            "weighted_rmse": self.weighted_rmse,
            "ridge_lambda": self.ridge_lambda,
            "condition_number": self.condition_number,
            "eig_range_unclipped": self.eig_range_unclipped,
            "eig_range_clipped": self.eig_range_clipped,
            "locality_bandwidth": self.locality_bandwidth,
            "reason": self.reason,
        }


def fit_local_surrogate(x_k, observations: list, cfg: CubicConfig) -> SurrogateFit:
    """Fit q_k(z) = g^T z + 0.5 z^T B z to y_i = F(p_i) - F(p_k).

    observations: list of (x_i ndarray, y_i float) measured on the SAME arm,
    model, task, metric definition, and search split as the incumbent (the
    caller enforces that identity; see DiscreteARCController.context_id).

    Steps: keep <= cfg.window nearest non-duplicate points by ||s_i||; build
    the local subspace from the SVD of the step matrix; choose the largest
    rank r <= cfg.max_rank with n >= r + r(r+1)/2 + 2; fall back to a linear
    ridge model (B=0) at n >= 3; otherwise return kind="none" (warm-up).
    Weighted ridge (lambda = cfg.ridge_lambda) is solved by lstsq on the
    augmented system; B is symmetrized and its eigenvalues clipped to
    [-cfg.eig_clip, +cfg.eig_clip].
    """
    np = _require_numpy()
    steps = []
    for x_i, y_i in observations:
        s = np.asarray(x_i, dtype=np.float64) - x_k
        nrm = float(np.linalg.norm(s))
        if nrm <= 1e-12:            # duplicate of the incumbent: carries no step info
            continue
        steps.append((nrm, s, float(y_i)))
    if not steps:
        return SurrogateFit(kind="none", n_observations=0,
                            ridge_lambda=cfg.ridge_lambda,
                            reason="no non-duplicate observations")

    steps.sort(key=lambda t: t[0])
    steps = steps[: cfg.window]
    n = len(steps)
    norms = np.array([t[0] for t in steps])
    S = np.stack([t[1] for t in steps])          # (n, d)
    y = np.array([t[2] for t in steps])

    # locality weights
    nonzero = norms[norms > 0]
    h = max(float(np.median(nonzero)) if nonzero.size else 0.0, 1e-3)
    a = np.exp(-0.5 * (norms / h) ** 2)

    # local subspace from the SVD of the step matrix
    try:
        _, sv, Vt = np.linalg.svd(S, full_matrices=False)
    except np.linalg.LinAlgError:
        return SurrogateFit(kind="none", n_observations=n,
                            ridge_lambda=cfg.ridge_lambda,
                            locality_bandwidth=h, reason="SVD failed")
    n_pos_sv = int((sv > 1e-12).sum())

    def _n_params_quad(r: int) -> int:
        return r + r * (r + 1) // 2

    rank_quad = 0
    for r in range(min(cfg.max_rank, n_pos_sv), 0, -1):
        if n >= _n_params_quad(r) + 2:
            rank_quad = r
            break

    if rank_quad == 0:
        # linear fallback (B = 0) once >= 3 distinct observations exist
        if n < 3:
            return SurrogateFit(kind="none", n_observations=n,
                                ridge_lambda=cfg.ridge_lambda,
                                locality_bandwidth=h,
                                reason=f"warm-up: n={n} < 3 observations")
        rank_lin = 0
        for r in range(min(cfg.max_rank, n_pos_sv), 0, -1):
            if n >= r + 2:
                rank_lin = r
                break
        if rank_lin == 0:
            return SurrogateFit(kind="none", n_observations=n,
                                ridge_lambda=cfg.ridge_lambda,
                                locality_bandwidth=h,
                                reason="no usable subspace rank")
        Q = Vt[:rank_lin].T                     # (d, r)
        Z = S @ Q                               # (n, r)
        design = Z
        p = rank_lin
        sqrt_a = np.sqrt(a)
        A_aug = np.vstack([design * sqrt_a[:, None],
                           math.sqrt(cfg.ridge_lambda) * np.eye(p)])
        y_aug = np.concatenate([y * sqrt_a, np.zeros(p)])
        theta, _, _, svals = np.linalg.lstsq(A_aug, y_aug, rcond=None)
        g = theta
        B = np.zeros((p, p))
        pred = design @ theta
        rmse = float(np.sqrt(np.sum(a * (pred - y) ** 2) / max(np.sum(a), 1e-12)))
        cond = float(svals[0] / svals[-1]) if (len(svals) and svals[-1] > 0) else None
        return SurrogateFit(
            kind="linear", rank=rank_lin, n_observations=n, g=g, B=B, Q=Q,
            weighted_rmse=rmse, ridge_lambda=cfg.ridge_lambda,
            condition_number=cond, eig_range_unclipped=[0.0, 0.0],
            eig_range_clipped=[0.0, 0.0], locality_bandwidth=h,
            reason="quadratic rank condition unmet; linear ridge with B=0")

    r = rank_quad
    Q = Vt[:r].T                                # (d, r)
    Z = S @ Q                                   # (n, r)

    # design: [ z_j | 0.5 z_j^2 | z_j z_l (j<l) ]
    cols = [Z]
    cols.append(0.5 * Z ** 2)
    off = []
    for j in range(r):
        for k2 in range(j + 1, r):
            off.append((Z[:, j] * Z[:, k2])[:, None])
    if off:
        cols.append(np.concatenate(off, axis=1))
    design = np.concatenate(cols, axis=1)
    p = design.shape[1]

    sqrt_a = np.sqrt(a)
    A_aug = np.vstack([design * sqrt_a[:, None],
                       math.sqrt(cfg.ridge_lambda) * np.eye(p)])
    y_aug = np.concatenate([y * sqrt_a, np.zeros(p)])
    theta, _, _, svals = np.linalg.lstsq(A_aug, y_aug, rcond=None)

    g = theta[:r].copy()
    B = np.zeros((r, r))
    for j in range(r):
        B[j, j] = theta[r + j]
    idx = 2 * r
    for j in range(r):
        for k2 in range(j + 1, r):
            B[j, k2] = theta[idx]
            B[k2, j] = theta[idx]
            idx += 1
    B = 0.5 * (B + B.T)                          # symmetrize (numerical hygiene)

    w, V = np.linalg.eigh(B)
    eig_raw = [float(w.min()), float(w.max())]
    w_clipped = np.clip(w, -cfg.eig_clip, cfg.eig_clip)
    B = (V * w_clipped) @ V.T
    eig_clipped = [float(w_clipped.min()), float(w_clipped.max())]

    pred = design @ theta
    rmse = float(np.sqrt(np.sum(a * (pred - y) ** 2) / max(np.sum(a), 1e-12)))
    cond = float(svals[0] / svals[-1]) if (len(svals) and svals[-1] > 0) else None

    return SurrogateFit(
        kind="quadratic", rank=r, n_observations=n, g=g, B=B, Q=Q,
        weighted_rmse=rmse, ridge_lambda=cfg.ridge_lambda,
        condition_number=cond, eig_range_unclipped=eig_raw,
        eig_range_clipped=eig_clipped, locality_bandwidth=h, reason="")


def cubic_model_value(fit: SurrogateFit, F_k: float, s, sigma: float) -> float:
    """m_k(p) = F(p_k) + g^T z + 0.5 z^T B z + (sigma/3) ||s||^3.

    ||s|| is the FULL feature-space step norm (penalizes directions outside
    the fitted subspace). With kind="none", the quadratic part is zero and
    the model is F_k + (sigma/3)||s||^3.
    """
    np = _require_numpy()
    s = np.asarray(s, dtype=np.float64)
    step_norm = float(np.linalg.norm(s))
    val = F_k + (sigma / 3.0) * step_norm ** 3
    if fit.kind in ("quadratic", "linear") and fit.Q is not None:
        z = fit.Q.T @ s
        val += float(fit.g @ z)
        if fit.kind == "quadratic":
            val += 0.5 * float(z @ fit.B @ z)
    return val


def compute_rho(ared: float, pred: float, pred_floor: float = 1e-8
                ) -> tuple[Optional[float], str]:
    """rho = ared/pred. If pred <= pred_floor the ratio is undefined for
    acceptance purposes: return (None, reason) — callers must treat this as
    'reject' (the classical rho = -inf case) and must serialize null + reason,
    never Infinity."""
    if pred <= pred_floor:
        return None, f"pred={pred:.3e} <= floor={pred_floor:.0e} (no predicted descent)"
    return ared / pred, ""


def sigma_update(sigma: float, rho: Optional[float], cfg: CubicConfig
                 ) -> tuple[float, str, bool]:
    """Return (new_sigma, outcome_label, accepted) per the D-ARC rules.

    very successful: rho >= eta2      -> accept, sigma = max(s_min, gamma_dec*s)
    successful:      eta1 <= rho<eta2 -> accept, sigma unchanged
    unsuccessful:    rho < eta1 or rho is None (pred<=floor / invalid / failed)
                                      -> reject, sigma = min(s_max, gamma_inc*s)
    """
    if rho is not None and rho >= cfg.eta2:
        return max(cfg.sigma_min, cfg.gamma_dec * sigma), "very_successful", True
    if rho is not None and rho >= cfg.eta1:
        return sigma, "successful", True
    return min(cfg.sigma_max, cfg.gamma_inc * sigma), "unsuccessful", False


# ---------------------------------------------------------------------------
# Proposals and records
# ---------------------------------------------------------------------------

@dataclass
class PromptProposal:
    """One candidate from the proposal generator. `rationale` is diagnostic
    metadata only and must NEVER be injected into the task agent's prompt."""
    prompt: str
    requested_scale: str = "medium"      # micro | small | medium | large
    rationale: str = ""


@dataclass
class CubicProposalScore:
    """Audit record for one proposal at one iteration."""
    prompt_sha: str
    requested_scale: str
    step_norm: Optional[float] = None
    model_value: Optional[float] = None
    predicted_reduction: Optional[float] = None
    selected: bool = False
    dropped_reason: str = ""             # "", "exact_duplicate", "feature_duplicate",
                                         # "already_evaluated", "invalid"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class CubicIterationRecord:
    """Complete, serializable audit record of one D-ARC iteration."""
    iteration: int
    phase: str                            # warmup | cubic | exploration | error
    sigma_before: float
    sigma_after: float
    target_radius: float
    prompt_sha: str = ""
    requested_scale: str = ""
    step_norm: Optional[float] = None
    model_value: Optional[float] = None
    predicted_reduction: Optional[float] = None
    actual_reduction: Optional[float] = None
    rho: Optional[float] = None           # None + rho_reason instead of +/-inf
    rho_reason: str = ""
    outcome: str = ""                     # very_successful | successful | unsuccessful
                                          # | warmup_improved | warmup_kept
                                          # | exploration | eval_failure
    accepted: bool = False
    F_incumbent_before: Optional[float] = None
    F_candidate: Optional[float] = None
    U_candidate: Optional[float] = None
    surrogate: dict = field(default_factory=dict)
    proposal_scores: list = field(default_factory=list)
    n_proposals_received: int = 0
    n_proposals_valid: int = 0
    consecutive_failures_after: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["proposal_scores"] = [
            p.to_dict() if isinstance(p, CubicProposalScore) else p
            for p in self.proposal_scores
        ]
        return d


_SCALE_ORDER = ["micro", "small", "medium", "large"]


# ---------------------------------------------------------------------------
# The controller
# ---------------------------------------------------------------------------

class DiscreteARCController:
    """State machine for D-ARC over a finite, LLM-proposed candidate stream.

    The controller never talks to an LLM or the filesystem. Per iteration the
    caller: (1) asks ``target_radius()`` and generates proposals, (2) calls
    ``select_proposal(proposals)``, (3) evaluates the returned prompt on the
    FIXED search split, (4) calls ``observe(prompt, F, U)`` (or
    ``observe_failure()`` if evaluation crashed). ``should_stop`` becomes True
    after ``cfg.patience`` consecutive rejected / no-descent iterations
    (post-warm-up).

    context_id encodes (arm, model, task, metric definition, split identity).
    Observations from a different context are refused — surrogate fits must
    never mix measurements from different minibatches, models, splits, or
    scoring configurations.
    """

    def __init__(self, cfg: CubicConfig, context_id: str,
                 feature_map: Optional[PromptFeatureMap] = None):
        _require_numpy()
        self.cfg = cfg
        self.context_id = context_id
        self.fmap = feature_map or PromptFeatureMap(cfg.hash_dims, cfg.ngram_sizes)
        self.sigma = cfg.sigma0
        self.iteration = 0
        self.consecutive_failures = 0
        self.should_stop = False
        self.records: list[CubicIterationRecord] = []
        self.incumbent_history: list[dict] = []
        # observations: prompt_sha -> (x, F, U); insertion-ordered
        self._obs: dict[str, tuple] = {}
        self._baseline: Optional[dict] = None
        self._incumbent: Optional[dict] = None
        self._accepted_incumbents: list[dict] = []
        self._pending: Optional[dict] = None      # selection awaiting observe()
        self._warmup_scale_idx = 0

    # -- helpers -----------------------------------------------------------
    def _sha(self, prompt: str) -> str:
        return self.fmap.prompt_key(prompt)

    @property
    def n_evaluations(self) -> int:
        """Evaluated candidates observed so far (excludes the baseline)."""
        return max(0, len(self._obs) - 1)

    @property
    def in_warmup(self) -> bool:
        return self.n_evaluations < self.cfg.warmup_evals

    def register_baseline(self, prompt: str, F: float, U: float,
                          context_id: Optional[str] = None) -> None:
        if context_id is not None and context_id != self.context_id:
            raise ValueError(
                f"context mismatch: controller={self.context_id!r} "
                f"baseline={context_id!r} — evaluation histories must not mix")
        sha = self._sha(prompt)
        x = self.fmap.features(prompt)
        self._obs[sha] = (x, float(F), float(U))
        self._baseline = {"prompt": prompt, "sha": sha, "F": float(F), "U": float(U)}
        self._incumbent = dict(self._baseline)
        self._accepted_incumbents = [dict(self._baseline)]
        self.incumbent_history.append(
            {"iteration": 0, "sha": sha, "F": float(F), "U": float(U),
             "event": "baseline"})

    @property
    def incumbent(self) -> dict:
        if self._incumbent is None:
            raise RuntimeError("register_baseline() must be called first")
        return self._incumbent

    def best_incumbent(self) -> dict:
        """Best ACCEPTED incumbent by scalar utility (the cubic-mode winner)."""
        return max(self._accepted_incumbents, key=lambda r: r["U"])

    def target_radius(self) -> float:
        return edit_radius(self.sigma)

    # -- selection -----------------------------------------------------------
    def _dedupe(self, proposals: list[PromptProposal]
                ) -> tuple[list[tuple[PromptProposal, str, Any]], list[CubicProposalScore]]:
        np = _np
        valid: list[tuple[PromptProposal, str, Any]] = []
        scores: list[CubicProposalScore] = []
        seen_shas: set[str] = set()
        for p in proposals:
            text = (p.prompt or "").strip()
            sha = self._sha(text) if text else ""
            rec = CubicProposalScore(prompt_sha=sha, requested_scale=p.requested_scale)
            if len(text) < 20:
                rec.dropped_reason = "invalid"
                scores.append(rec)
                continue
            if sha in seen_shas:
                rec.dropped_reason = "exact_duplicate"
                scores.append(rec)
                continue
            if sha in self._obs:
                rec.dropped_reason = "already_evaluated"
                scores.append(rec)
                continue
            x = self.fmap.features(text)
            dup = False
            for _, prev_sha, prev_x in valid:
                if float(np.linalg.norm(x - prev_x)) < self.cfg.dedup_feature_eps:
                    dup = True
                    break
            if dup:
                rec.dropped_reason = "feature_duplicate"
                scores.append(rec)
                continue
            seen_shas.add(sha)
            valid.append((p, sha, x))
            scores.append(rec)
        return valid, scores

    def select_proposal(self, proposals: list[PromptProposal]) -> dict:
        """Choose which proposal to evaluate this iteration.

        Returns {"prompt", "sha", "phase", "scores", "surrogate", "no_descent"}.
        phase: "warmup" (deterministic scale cycling), "cubic" (min m_k), or
        "exploration" (post-warm-up, no proposal with positive predicted
        reduction: caller must still evaluate the returned smallest-step
        proposal as an exploration observation; sigma has already been
        increased). Returns {"prompt": None, ...} if nothing valid remains —
        the caller should retry the proposal call once, then call
        ``observe_failure`` if still empty.
        Ties on the cubic model value break by (1) smaller full step norm,
        (2) lexicographic SHA-256 of the prompt.
        """
        np = _np
        inc = self.incumbent
        x_k = self.fmap.features(inc["prompt"])
        valid, scores = self._dedupe(proposals)
        by_sha = {sha: rec for rec, (_, sha, _x) in
                  zip([s for s in scores if not s.dropped_reason], valid)}

        result = {
            "prompt": None, "sha": "", "requested_scale": "", "phase": "",
            "scores": scores, "surrogate": {}, "no_descent": False,
        }
        if not valid:
            return result

        # step norms for every valid proposal
        for prop, sha, x in valid:
            s = x - x_k
            by_sha[sha].step_norm = float(np.linalg.norm(s))

        if self.in_warmup:
            # deterministic cycling over available requested edit scales
            want = _SCALE_ORDER[self._warmup_scale_idx % len(_SCALE_ORDER)]
            ordered = sorted(
                valid,
                key=lambda t: (
                    0 if t[0].requested_scale == want else
                    1 + _SCALE_ORDER.index(t[0].requested_scale)
                    if t[0].requested_scale in _SCALE_ORDER else 9,
                    by_sha[t[1]].step_norm if by_sha[t[1]].step_norm is not None else 0.0,
                    t[1],
                ),
            )
            prop, sha, x = ordered[0]
            self._warmup_scale_idx += 1
            by_sha[sha].selected = True
            self._pending = {
                "sha": sha, "prompt": prop.prompt, "scale": prop.requested_scale,
                "phase": "warmup", "pred": None, "model_value": None,
                "step_norm": by_sha[sha].step_norm,
                "surrogate": {}, "scores": scores,
                "n_received": len(proposals), "n_valid": len(valid),
            }
            result.update(prompt=prop.prompt, sha=sha,
                          requested_scale=prop.requested_scale, phase="warmup")
            return result

        # post-warm-up: fit the surrogate and minimize the cubic model
        obs = [(x, F) for sha_o, (x, F, _U) in self._obs.items()]
        F_k = inc["F"]
        fit = fit_local_surrogate(
            x_k, [(x, F - F_k) for (x, F) in obs], self.cfg)
        result["surrogate"] = fit.diagnostics()

        best = None            # (model_value, step_norm, sha, prop, pred)
        for prop, sha, x in valid:
            s = x - x_k
            m = cubic_model_value(fit, F_k, s, self.sigma)
            pred = F_k - m
            rec = by_sha[sha]
            rec.model_value = m
            rec.predicted_reduction = pred
            key = (m, rec.step_norm, sha)
            if best is None or key < (best[0], best[1], best[2]):
                best = (m, rec.step_norm, sha, prop, pred)

        m, step_norm, sha, prop, pred = best
        if pred <= self.cfg.pred_floor:
            # no predicted descent anywhere: increase sigma now; hand back the
            # smallest-step proposal as an exploration observation only.
            self.sigma = min(self.cfg.sigma_max, self.cfg.gamma_inc * self.sigma)
            explore = min(valid, key=lambda t: (by_sha[t[1]].step_norm, t[1]))
            e_prop, e_sha, _ = explore
            by_sha[e_sha].selected = True
            self._pending = {
                "sha": e_sha, "prompt": e_prop.prompt,
                "scale": e_prop.requested_scale, "phase": "exploration",
                "pred": by_sha[e_sha].predicted_reduction,
                "model_value": by_sha[e_sha].model_value,
                "step_norm": by_sha[e_sha].step_norm,
                "surrogate": fit.diagnostics(), "scores": scores,
                "n_received": len(proposals), "n_valid": len(valid),
            }
            result.update(prompt=e_prop.prompt, sha=e_sha,
                          requested_scale=e_prop.requested_scale,
                          phase="exploration", no_descent=True)
            return result

        by_sha[sha].selected = True
        self._pending = {
            "sha": sha, "prompt": prop.prompt, "scale": prop.requested_scale,
            "phase": "cubic", "pred": pred, "model_value": m,
            "step_norm": step_norm, "surrogate": fit.diagnostics(),
            "scores": scores, "n_received": len(proposals), "n_valid": len(valid),
        }
        result.update(prompt=prop.prompt, sha=sha,
                      requested_scale=prop.requested_scale, phase="cubic")
        return result

    # -- observation ---------------------------------------------------------
    def observe(self, prompt: str, F: float, U: float,
                context_id: Optional[str] = None) -> CubicIterationRecord:
        """Record the evaluation of the prompt returned by select_proposal."""
        if context_id is not None and context_id != self.context_id:
            raise ValueError(
                f"context mismatch: controller={self.context_id!r} "
                f"observation={context_id!r} — evaluation histories must not mix")
        if self._pending is None:
            raise RuntimeError("observe() called without a pending selection")
        pend = self._pending
        self._pending = None
        sha = self._sha(prompt)
        if sha != pend["sha"]:
            raise ValueError("observe() got a different prompt than was selected")

        self.iteration += 1
        inc = self.incumbent
        sigma_before = self.sigma
        x = self.fmap.features(prompt)
        self._obs[sha] = (x, float(F), float(U))

        rec = CubicIterationRecord(
            iteration=self.iteration, phase=pend["phase"],
            sigma_before=sigma_before, sigma_after=self.sigma,
            target_radius=edit_radius(sigma_before),
            prompt_sha=sha, requested_scale=pend["scale"],
            step_norm=pend["step_norm"], model_value=pend["model_value"],
            predicted_reduction=pend["pred"],
            F_incumbent_before=inc["F"], F_candidate=float(F),
            U_candidate=float(U), surrogate=pend["surrogate"],
            proposal_scores=pend["scores"],
            n_proposals_received=pend["n_received"],
            n_proposals_valid=pend["n_valid"],
        )

        if pend["phase"] == "warmup":
            # incumbent moves only on STRICT measured scalar-utility improvement
            if U > inc["U"] + self.cfg.utility_improve_eps:
                self._move_incumbent(prompt, sha, F, U, "warmup_improved")
                rec.outcome, rec.accepted = "warmup_improved", True
            else:
                rec.outcome, rec.accepted = "warmup_kept", False
            rec.sigma_after = self.sigma           # sigma untouched in warm-up
            rec.consecutive_failures_after = self.consecutive_failures
            self.records.append(rec)
            return rec

        if pend["phase"] == "exploration":
            # sigma was already increased at selection; never accepted as a step
            rec.outcome, rec.accepted = "exploration", False
            rec.rho_reason = "exploration observation (no positive predicted reduction)"
            rec.sigma_after = self.sigma
            self.consecutive_failures += 1
            rec.consecutive_failures_after = self.consecutive_failures
            if self.consecutive_failures >= self.cfg.patience:
                self.should_stop = True
                rec.note = f"stop: {self.consecutive_failures} consecutive non-descent iterations"
            self.records.append(rec)
            return rec

        # phase == "cubic"
        ared = inc["F"] - float(F)
        rec.actual_reduction = ared
        rho, reason = compute_rho(ared, pend["pred"] if pend["pred"] is not None else 0.0,
                                  self.cfg.pred_floor)
        rec.rho, rec.rho_reason = rho, reason
        self.sigma, outcome, accepted = sigma_update(self.sigma, rho, self.cfg)
        rec.sigma_after = self.sigma
        rec.outcome, rec.accepted = outcome, accepted

        if accepted:
            self._move_incumbent(prompt, sha, F, U, outcome)
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.cfg.patience:
                self.should_stop = True
                rec.note = f"stop: {self.consecutive_failures} consecutive rejected iterations"
        rec.consecutive_failures_after = self.consecutive_failures
        self.records.append(rec)
        return rec

    def observe_failure(self, reason: str = "evaluation failed") -> CubicIterationRecord:
        """An evaluation crash: sigma increases, state stays consistent."""
        self.iteration += 1
        sigma_before = self.sigma
        self.sigma = min(self.cfg.sigma_max, self.cfg.gamma_inc * self.sigma)
        pend = self._pending or {}
        self._pending = None
        self.consecutive_failures += 1
        rec = CubicIterationRecord(
            iteration=self.iteration, phase="error",
            sigma_before=sigma_before, sigma_after=self.sigma,
            target_radius=edit_radius(sigma_before),
            prompt_sha=pend.get("sha", ""), requested_scale=pend.get("scale", ""),
            outcome="eval_failure", accepted=False, rho=None, rho_reason=reason,
            surrogate=pend.get("surrogate", {}),
            proposal_scores=pend.get("scores", []),
            n_proposals_received=pend.get("n_received", 0),
            n_proposals_valid=pend.get("n_valid", 0),
            consecutive_failures_after=self.consecutive_failures,
            note=reason,
        )
        if self.consecutive_failures >= self.cfg.patience:
            self.should_stop = True
        self.records.append(rec)
        return rec

    def _move_incumbent(self, prompt: str, sha: str, F: float, U: float,
                        event: str) -> None:
        self._incumbent = {"prompt": prompt, "sha": sha, "F": float(F), "U": float(U)}
        self._accepted_incumbents.append(dict(self._incumbent))
        self.incumbent_history.append(
            {"iteration": self.iteration, "sha": sha, "F": float(F),
             "U": float(U), "event": event})

    # -- export ----------------------------------------------------------------
    def history_dicts(self) -> list[dict]:
        return [r.to_dict() for r in self.records]

    def summary(self) -> dict:
        acc = [r for r in self.records if r.accepted]
        rej = [r for r in self.records
               if not r.accepted and r.phase in ("cubic", "exploration", "error")]
        return {
            "context_id": self.context_id,
            "iterations": self.iteration,
            "accepted_steps": len(acc),
            "rejected_steps": len(rej),
            "final_sigma": self.sigma,
            "stopped_early": self.should_stop,
            "incumbent_sha": self._incumbent["sha"] if self._incumbent else "",
            "incumbent_U": self._incumbent["U"] if self._incumbent else None,
            "incumbent_F": self._incumbent["F"] if self._incumbent else None,
            "config": self.cfg.to_dict(),
        }


__all__ = [
    "CubicConfig", "PromptFeatureMap", "SurrogateFit", "fit_local_surrogate",
    "cubic_model_value", "compute_rho", "sigma_update", "edit_radius",
    "scalar_utility", "PromptProposal", "CubicProposalScore",
    "CubicIterationRecord", "DiscreteARCController",
]
