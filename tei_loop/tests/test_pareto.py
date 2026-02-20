"""Tests for the Pareto front management module."""
import pytest
from tei_loop.pareto import is_dominated, update_pareto_front, sample_from_front, sample_pair_from_front, compute_composite, select_best
from tei_loop.models import ParetoCandidate


def _make_candidate(scores: dict, composite: float = 0.0, **kwargs) -> ParetoCandidate:
    return ParetoCandidate(
        iteration=kwargs.get("iteration", 0),
        prompt_text=kwargs.get("prompt", "test"),
        metric_scores=scores,
        composite_score=composite,
    )


class TestIsDominated:
    def test_dominated(self):
        a = _make_candidate({"m1": 50, "m2": 60})
        b = _make_candidate({"m1": 70, "m2": 80})
        assert is_dominated(a, b)

    def test_not_dominated_equal(self):
        a = _make_candidate({"m1": 70, "m2": 80})
        b = _make_candidate({"m1": 70, "m2": 80})
        assert not is_dominated(a, b)

    def test_not_dominated_tradeoff(self):
        a = _make_candidate({"m1": 90, "m2": 50})
        b = _make_candidate({"m1": 50, "m2": 90})
        assert not is_dominated(a, b)


class TestUpdateParetoFront:
    def test_add_non_dominated(self):
        front = [_make_candidate({"m1": 50, "m2": 60}, composite=55)]
        new = _make_candidate({"m1": 70, "m2": 50}, composite=60)
        result = update_pareto_front(front, new)
        assert len(result) == 2

    def test_remove_dominated(self):
        front = [_make_candidate({"m1": 50, "m2": 60}, composite=55)]
        new = _make_candidate({"m1": 70, "m2": 80}, composite=75)
        result = update_pareto_front(front, new)
        assert len(result) == 1
        assert result[0].composite_score == 75


class TestSampling:
    def test_sample_single(self):
        front = [_make_candidate({"m1": 50}, composite=50)]
        result = sample_from_front(front)
        assert result.composite_score == 50

    def test_sample_pair(self):
        front = [
            _make_candidate({"m1": 50}, composite=50),
            _make_candidate({"m1": 70}, composite=70),
        ]
        a, b = sample_pair_from_front(front)
        assert a.candidate_id != b.candidate_id

    def test_sample_empty_raises(self):
        with pytest.raises(ValueError):
            sample_from_front([])

    def test_sample_pair_too_few_raises(self):
        with pytest.raises(ValueError):
            sample_pair_from_front([_make_candidate({"m1": 50})])


class TestComposite:
    def test_compute(self):
        scores = {"m1": 0.8, "m2": 0.6}
        weights = {"m1": 0.5, "m2": 0.5}
        assert compute_composite(scores, weights) == 70.0


class TestSelectBest:
    def test_select(self):
        front = [
            _make_candidate({"m1": 50}, composite=50),
            _make_candidate({"m1": 90}, composite=90),
            _make_candidate({"m1": 70}, composite=70),
        ]
        best = select_best(front)
        assert best.composite_score == 90

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            select_best([])
