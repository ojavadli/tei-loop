"""Tests for the paper's Algorithm-1 mechanics: clamp and the do-no-harm gate."""

import pytest

from tei_loop.gate import EPS, GateDecision, do_no_harm, sign_test_p
from tei_loop.models import clamp


def test_clamp_paper_eq1():
    assert clamp(1.5) == 0.999
    assert clamp(1.0) == 0.999
    assert clamp(0.999) == 0.999
    assert clamp(0.5) == 0.5
    assert clamp(0.0) == 0.0
    assert clamp(-0.3) == 0.0


def test_sign_test_exact_values():
    # p = min(1, 2 * sum_{j<=min(W,L)} C(m,j) / 2^m)
    assert sign_test_p(5, 0) == pytest.approx(2 * 1 / 32)
    assert sign_test_p(4, 1) == pytest.approx(2 * (1 + 5) / 32)
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(3, 3) == 1.0  # min(1, .) cap engaged


def test_gate_accepts_on_wins_and_mean():
    d = do_no_harm([0.8, 0.9, 0.7], [0.7, 0.9, 0.6])
    assert d.accept and (d.wins, d.losses, d.ties) == (2, 0, 1)


def test_gate_rejects_when_losses_exceed_wins():
    d = do_no_harm([0.5, 0.5, 0.9], [0.6, 0.6, 0.7])
    assert not d.accept and d.losses == 2 and d.wins == 1


def test_gate_rejects_on_mean_even_with_equal_wins():
    # one big loss vs one small win: W == L but mean drops -> reject
    d = do_no_harm([0.9, 0.1], [0.85, 0.6])
    assert d.wins == d.losses == 1 and not d.accept


def test_gate_tie_only_is_accepted_as_no_harm():
    d = do_no_harm([0.7, 0.7], [0.7, 0.7])
    assert d.accept and d.ties == 2 and d.p_sign == 1.0


def test_gate_eps_treats_noise_as_tie():
    d = do_no_harm([0.7 + EPS / 2], [0.7])
    assert d.ties == 1 and d.wins == 0


def test_gate_requires_paired_inputs():
    with pytest.raises(ValueError):
        do_no_harm([0.5], [0.5, 0.6])
    with pytest.raises(ValueError):
        do_no_harm([], [])


def test_gate_summary_is_reportable():
    s = do_no_harm([0.8], [0.7]).summary()
    assert "ACCEPT" in s and "W/L/T 1/0/0" in s
