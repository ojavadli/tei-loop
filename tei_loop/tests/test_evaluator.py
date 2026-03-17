"""Tests for TEI evaluator orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from tei_loop.models import (
    Dimension,
    DimensionConfig,
    DimensionScore,
    TEIConfig,
    Trace,
    TraceStep,
)
from tei_loop.evaluator import TEIEvaluator


def _make_trace(agent_input="test query", agent_output="test response"):
    return Trace(
        agent_input=agent_input,
        agent_output=agent_output,
        steps=[TraceStep(name="agent_run", input_data=agent_input, output_data=agent_output)],
    )


def _make_mock_llm():
    mock = AsyncMock()
    mock.generate_json = AsyncMock(return_value={
        "score": 0.85,
        "assertions": [
            {"claim": "Test assertion", "evidence": "Test evidence", "verdict": "pass", "explanation": "OK"}
        ],
        "reasoning": "Good performance",
        "failure_summary": None,
    })
    mock.get_cost = MagicMock(return_value=0.001)
    return mock


@pytest.mark.asyncio
async def test_evaluator_returns_all_dimensions():
    config = TEIConfig()
    mock_llm = _make_mock_llm()
    evaluator = TEIEvaluator(config, mock_llm)

    trace = _make_trace()
    result = await evaluator.evaluate(trace)

    assert len(result.dimension_scores) == 4
    for dim in Dimension:
        assert dim in result.dimension_scores


@pytest.mark.asyncio
async def test_evaluator_aggregate_score():
    config = TEIConfig()
    mock_llm = _make_mock_llm()
    evaluator = TEIEvaluator(config, mock_llm)

    trace = _make_trace()
    result = await evaluator.evaluate(trace)

    assert 0 <= result.aggregate_score <= 0.97
    assert result.aggregate_score == pytest.approx(0.85, abs=0.01)


@pytest.mark.asyncio
async def test_evaluator_all_passed():
    config = TEIConfig()
    mock_llm = _make_mock_llm()
    evaluator = TEIEvaluator(config, mock_llm)

    trace = _make_trace()
    result = await evaluator.evaluate(trace)

    assert result.all_passed is True


@pytest.mark.asyncio
async def test_evaluator_failure_detection():
    config = TEIConfig()
    mock_llm = AsyncMock()
    mock_llm.generate_json = AsyncMock(return_value={
        "score": 0.45,
        "assertions": [
            {"claim": "Failed check", "evidence": "Bad output", "verdict": "fail", "explanation": "Wrong"}
        ],
        "reasoning": "Poor performance",
        "failure_summary": "Agent missed the target",
    })
    mock_llm.get_cost = MagicMock(return_value=0.001)

    evaluator = TEIEvaluator(config, mock_llm)
    trace = _make_trace()
    result = await evaluator.evaluate(trace)

    assert result.all_passed is False
    failures = result.get_failures()
    assert len(failures) == 4


@pytest.mark.asyncio
async def test_evaluator_disabled_dimension():
    config = TEIConfig()
    config.dimensions[Dimension.EXECUTION_ACCURACY] = DimensionConfig(enabled=False)
    mock_llm = _make_mock_llm()
    evaluator = TEIEvaluator(config, mock_llm)

    trace = _make_trace()
    result = await evaluator.evaluate(trace)

    assert Dimension.EXECUTION_ACCURACY not in result.dimension_scores
    assert len(result.dimension_scores) == 3


@pytest.mark.asyncio
async def test_evaluator_handles_llm_error():
    config = TEIConfig()
    mock_llm = AsyncMock()
    mock_llm.generate_json = AsyncMock(side_effect=Exception("API error"))
    mock_llm.get_cost = MagicMock(return_value=0)

    evaluator = TEIEvaluator(config, mock_llm)
    trace = _make_trace()
    result = await evaluator.evaluate(trace)

    assert result.aggregate_score == 0.0
    assert result.all_passed is False
    for ds in result.dimension_scores.values():
        assert "failed" in ds.reasoning.lower() or "error" in ds.reasoning.lower()


@pytest.mark.asyncio
async def test_evaluate_single_dimension():
    config = TEIConfig()
    mock_llm = _make_mock_llm()
    evaluator = TEIEvaluator(config, mock_llm)

    trace = _make_trace()
    score = await evaluator.evaluate_single(trace, Dimension.TARGET_ALIGNMENT)

    assert score.dimension == Dimension.TARGET_ALIGNMENT
    assert score.score == pytest.approx(0.85, abs=0.01)
