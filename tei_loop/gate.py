"""v7 decision gate for TEI-loop: paired, noise-aware keep/apply decisions.

Why this exists (evidence: the tei-bench study, results v2-v6): with a handful
of evaluation queries, point-estimate comparisons of LLM-judge scores are
dominated by judge noise — "improvements" selected that way frequently fail to
generalize (7/31 held-out regressions in the naive loop), and no optimizer can
out-select the noise (v5). The product-grade remedies implemented here:

  * verify_candidate(): a paired do-no-harm rule over the FULL evaluation
    query set (the previous final check used a single query). Accept only if
    the candidate's mean is not below the reference AND it does not lose on
    more queries than it wins. The exact two-sided sign-test p-value over
    discordant queries is reported alongside, as evidence.
  * preflight_power(): tells the user, before optimization spends anything,
    the smallest score change that is statistically meaningful at their
    number of eval queries — so a "no certified improvement" outcome is
    interpretable, and so users add queries instead of trusting noise.

Pure standard library (math only), so it adds no dependencies to the product.
"""
from __future__ import annotations

import math

# two-sided z for alpha=0.05 and z for 80% power (normal approximation)
_Z_ALPHA = 1.96
_Z_POWER = 0.84


def hoeffding_margin(n: int, alpha: float = 0.05) -> float:
    """Distribution-free margin for a mean of n bounded [0,1] scores."""
    if n <= 0:
        return 1.0
    return math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


def exact_sign_p(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test on discordant queries."""
    m = wins + losses
    if m == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(m, j) for j in range(0, k + 1)) / (2.0 ** m)
    return min(1.0, 2.0 * tail)


def mde_queries(n: int, per_query_sd: float = 0.15) -> float:
    """Minimum detectable mean-score change at n paired queries (alpha=.05,
    power=.80), assuming per-query judge-score sd ~= per_query_sd (empirically
    reasonable for 0-1 aggregate judge scores; paired-diff sd = sd*sqrt(2))."""
    if n <= 0:
        return float("inf")
    return (_Z_ALPHA + _Z_POWER) * per_query_sd * math.sqrt(2.0) / math.sqrt(n)


def queries_needed(target_delta: float = 0.10, per_query_sd: float = 0.15) -> int:
    """How many eval queries certify an improvement of target_delta."""
    if target_delta <= 0:
        return 0
    n = (( _Z_ALPHA + _Z_POWER) * per_query_sd * math.sqrt(2.0) / target_delta) ** 2
    return max(1, math.ceil(n))


def preflight_power(n_queries: int) -> str:
    """Human-readable power warning printed before optimization starts."""
    mde = mde_queries(n_queries)
    need = queries_needed(0.10)
    if n_queries < 3:
        return (f"Power check: {n_queries} eval quer{'y' if n_queries == 1 else 'ies'} "
                f"- score differences are judge noise at this sample size; "
                f"improvements will be applied only if they never score below the "
                f"reference. Provide ~{need} queries to certify ~0.10 gains.")
    return (f"Power check: with {n_queries} eval queries, only mean-score changes "
            f">= ~{mde:.2f} are statistically meaningful; smaller deltas are "
            f"within judge noise. (~{need} queries would certify a 0.10 gain.)")


def verify_candidate(cand_scores: list[float], ref_scores: list[float],
                     tie_eps: float = 1e-9) -> dict:
    """Paired do-no-harm verdict on per-query aggregate scores.

    Accept iff (a) candidate mean >= reference mean, and (b) the candidate does
    not lose on more queries than it wins. With fewer than 3 queries the rule
    tightens to "no losses at all" (nothing is certifiable there; do no harm).
    Returns a dict with accept, reason, means, wins/losses, sign_p, margin, mde.
    """
    n = min(len(cand_scores), len(ref_scores))
    cand = [float(x) for x in cand_scores[:n]]
    ref = [float(x) for x in ref_scores[:n]]
    cand_mean = sum(cand) / n if n else 0.0
    ref_mean = sum(ref) / n if n else 0.0
    wins = sum(1 for c, r in zip(cand, ref) if c > r + tie_eps)
    losses = sum(1 for c, r in zip(cand, ref) if c < r - tie_eps)

    verdict = {"n_queries": n, "cand_mean": round(cand_mean, 4),
               "ref_mean": round(ref_mean, 4),
               "delta": round(cand_mean - ref_mean, 4),
               "wins": wins, "losses": losses,
               "sign_p": round(exact_sign_p(wins, losses), 3),
               "margin_hoeffding": round(hoeffding_margin(n), 3),
               "mde": round(mde_queries(n), 3),
               "insufficient_n": n < 3}

    if n == 0:
        verdict.update(accept=False, reason="no evaluation queries")
        return verdict
    if cand_mean < ref_mean - tie_eps:
        verdict.update(accept=False, reason="mean below reference")
        return verdict
    if n < 3:
        ok = losses == 0
        verdict.update(accept=ok,
                       reason="do-no-harm at tiny n: accepted with no losing query"
                       if ok else "a query regressed and n<3 certifies nothing")
        return verdict
    if losses > wins:
        verdict.update(accept=False, reason="loses on more queries than it wins")
        return verdict
    verdict.update(accept=True,
                   reason=f"mean not below reference; {wins}W/{losses}L paired "
                          f"(sign p={verdict['sign_p']})")
    return verdict


__all__ = ["hoeffding_margin", "exact_sign_p", "mde_queries", "queries_needed",
           "preflight_power", "verify_candidate"]


def static_pregate(path: str, new_text: str) -> dict:
    """Zero-cost deterministic pre-gate: reject a candidate whose patched file
    no longer parses, before any judge or agent call is spent on it.

    Motivated by the TEI-SWE study (30 SWE-bench leaderboard agents), where a
    post-hoc audit found 6 applied patches had introduced Python syntax errors
    that the rubric judge then scored as improvements. ``ast.parse`` catches
    that entire failure class for free. Non-Python files pass through.
    """
    if not path.endswith(".py"):
        return {"accept": True, "reason": "no static check for this file type"}
    import ast
    try:
        ast.parse(new_text)
        return {"accept": True, "reason": "parses"}
    except SyntaxError as e:
        return {"accept": False,
                "reason": f"SyntaxError line {e.lineno}: {e.msg}"}
