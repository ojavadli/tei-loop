"""Tests for TEI improver module."""

import pytest
from tei_loop.models import (
    Dimension,
    DimensionScore,
    EvalResult,
    FixStrategyType,
    Severity,
)
from tei_loop.improver import classify_failures, check_convergence


def test_classify_no_failures():
    er = EvalResult(
        dimension_scores={
            Dimension.TARGET_ALIGNMENT: DimensionScore(
                dimension=Dimension.TARGET_ALIGNMENT, score=0.85, passed=True, threshold=0.7
            ),
        },
        all_passed=True,
    )
    failures = classify_failures(er)
    assert len(failures) == 0


def test_classify_critical_failure():
    er = EvalResult(
        dimension_scores={
            Dimension.TARGET_ALIGNMENT: DimensionScore(
                dimension=Dimension.TARGET_ALIGNMENT,
                score=0.2,
                passed=False,
                threshold=0.7,
                failure_summary="Agent completely misunderstood the query",
            ),
        },
        all_passed=False,
    )
    failures = classify_failures(er)
    assert len(failures) == 1
    assert failures[0].severity == Severity.CRITICAL
    assert failures[0].suggested_strategy == FixStrategyType.TARGET_REANCHOR


def test_classify_severity_ordering():
    er = EvalResult(
        dimension_scores={
            Dimension.OUTPUT_INTEGRITY: DimensionScore(
                dimension=Dimension.OUTPUT_INTEGRITY,
                score=0.60,
                passed=False,
                threshold=0.7,
                failure_summary="Minor formatting issues",
            ),
            Dimension.TARGET_ALIGNMENT: DimensionScore(
                dimension=Dimension.TARGET_ALIGNMENT,
                score=0.15,
                passed=False,
                threshold=0.7,
                failure_summary="Completely wrong target",
            ),
        },
        all_passed=False,
    )
    failures = classify_failures(er)
    assert failures[0].severity == Severity.CRITICAL
    assert failures[0].dimension == Dimension.TARGET_ALIGNMENT


def test_strategy_mapping():
    er = EvalResult(
        dimension_scores={
            Dimension.REASONING_SOUNDNESS: DimensionScore(
                dimension=Dimension.REASONING_SOUNDNESS,
                score=0.40,
                passed=False,
                threshold=0.65,
                failure_summary="Contradictory reasoning",
            ),
            Dimension.EXECUTION_ACCURACY: DimensionScore(
                dimension=Dimension.EXECUTION_ACCURACY,
                score=0.50,
                passed=False,
                threshold=0.7,
                failure_summary="Wrong API parameters",
            ),
        },
        all_passed=False,
    )
    failures = classify_failures(er)
    strategies = {f.dimension: f.suggested_strategy for f in failures}
    assert strategies[Dimension.REASONING_SOUNDNESS] == FixStrategyType.REASONING_REGENERATE
    assert strategies[Dimension.EXECUTION_ACCURACY] == FixStrategyType.EXECUTION_CORRECT


def test_convergence_not_enough_data():
    assert not check_convergence([0.5])
    assert not check_convergence([0.5, 0.6])


def test_convergence_detected():
    assert check_convergence([0.5, 0.7, 0.71, 0.71, 0.72], threshold=0.02, window=3)


def test_convergence_not_detected():
    assert not check_convergence([0.5, 0.6, 0.7, 0.8, 0.85], threshold=0.02, window=3)
