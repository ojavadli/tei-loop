"""
The do-no-harm deployment gate (paper, Sec. 2.4 / Algorithm 1 line 14).

A candidate ships only if, over paired comparisons against the reference,
    mean(candidate) >= mean(reference)   AND   losses <= wins,
where a pair is a win/loss when the candidate is above/below the reference by
more than eps, and a tie otherwise. The exact two-sided sign-test p-value
    p = min(1, 2 * sum_{j=0..min(W,L)} C(m, j) / 2^m),   m = W + L,
is reported for transparency (the gate itself is the two conditions above).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb


EPS = 1e-9


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign-test p-value over the non-tied pairs, capped at 1."""
    m = wins + losses
    if m == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(m, j) for j in range(k + 1)) / (2 ** m)
    return min(1.0, 2.0 * tail)


@dataclass
class GateDecision:
    accept: bool
    mean_candidate: float
    mean_reference: float
    wins: int
    losses: int
    ties: int
    p_sign: float

    def summary(self) -> str:
        verdict = "ACCEPT" if self.accept else "KEEP REFERENCE"
        return (
            f"{verdict}: mean {self.mean_candidate:.3f} vs {self.mean_reference:.3f}, "
            f"W/L/T {self.wins}/{self.losses}/{self.ties}, sign-test p={self.p_sign:.3f}"
        )


def do_no_harm(candidate: list[float], reference: list[float], eps: float = EPS) -> GateDecision:
    """Apply the paper's gate to paired score lists (per probe query, or per
    dimension when only a single probe is available)."""
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("gate needs equal-length, non-empty paired scores")
    wins = sum(1 for c, r in zip(candidate, reference) if c > r + eps)
    losses = sum(1 for c, r in zip(candidate, reference) if c < r - eps)
    ties = len(candidate) - wins - losses
    mean_c = sum(candidate) / len(candidate)
    mean_r = sum(reference) / len(reference)
    return GateDecision(
        accept=(mean_c >= mean_r and losses <= wins),
        mean_candidate=mean_c,
        mean_reference=mean_r,
        wins=wins,
        losses=losses,
        ties=ties,
        p_sign=sign_test_p(wins, losses),
    )
