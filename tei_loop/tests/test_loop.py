"""Tests for TEI loop controller."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tei_loop.models import (
    Dimension,
    DimensionScore,
    EvalResult,
    TEIConfig,
    TEIFullResult,
    Trace,
    TraceStep,
)


def _mock_agent(query):
    return f"Response to: {query}"


def _make_eval_result(score=0.85, all_passed=True):
    dims = {}
    for d in Dimension:
        dims[d] = DimensionScore(
            dimension=d, score=score, passed=all_passed, threshold=0.7
        )
    return EvalResult(
        dimension_scores=dims,
        aggregate_score=score,
        all_passed=all_passed,
    )


@pytest.mark.asyncio
async def test_loop_init_with_defaults():
    with patch("tei_loop.loop.build_providers") as mock_build:
        mock_eval_llm = MagicMock()
        mock_improve_llm = MagicMock()
        mock_build.return_value = ("openai", mock_eval_llm, mock_improve_llm)

        from tei_loop import TEILoop
        loop = TEILoop(agent=_mock_agent)
        assert loop.config is not None
        assert loop._verbose is True
        assert loop._interactive is True


@pytest.mark.asyncio
async def test_loop_evaluate_only():
    with patch("tei_loop.loop.build_providers") as mock_build, \
         patch("tei_loop.loop.run_and_trace") as mock_trace:

        mock_eval_llm = MagicMock()
        mock_eval_llm.get_cost = MagicMock(return_value=0.01)
        mock_improve_llm = MagicMock()
        mock_improve_llm.get_cost = MagicMock(return_value=0.0)
        mock_build.return_value = ("openai", mock_eval_llm, mock_improve_llm)

        trace = Trace(
            agent_input="test",
            agent_output="response",
            steps=[TraceStep(name="run")],
        )
        mock_trace.return_value = trace

        from tei_loop import TEILoop
        loop = TEILoop(agent=_mock_agent)
        loop._initialized = True
        loop._provider_name = "openai"
        loop._eval_llm = mock_eval_llm
        loop._improve_llm = mock_improve_llm

        from tei_loop.evaluator import TEIEvaluator
        mock_evaluator = MagicMock(spec=TEIEvaluator)
        mock_evaluator.evaluate = AsyncMock(
            return_value=_make_eval_result(0.85, True)
        )
        loop._evaluator = mock_evaluator

        result = await loop.evaluate_only("test query")

        assert isinstance(result, EvalResult)
        assert result.aggregate_score == 0.85
        assert result.all_passed is True


@pytest.mark.asyncio
async def test_evaluate_only_calls_trace_and_eval():
    with patch("tei_loop.loop.build_providers") as mock_build, \
         patch("tei_loop.loop.run_and_trace") as mock_trace:

        mock_eval_llm = MagicMock()
        mock_improve_llm = MagicMock()
        mock_build.return_value = ("openai", mock_eval_llm, mock_improve_llm)

        trace = Trace(agent_input="q", agent_output="a")
        mock_trace.return_value = trace

        from tei_loop import TEILoop
        loop = TEILoop(agent=_mock_agent)
        loop._initialized = True
        loop._provider_name = "openai"
        loop._eval_llm = mock_eval_llm
        loop._improve_llm = mock_improve_llm

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value=_make_eval_result(0.60, False)
        )
        loop._evaluator = mock_evaluator

        result = await loop.evaluate_only("test")

        mock_trace.assert_called_once()
        mock_evaluator.evaluate.assert_called_once()
        assert result.aggregate_score == 0.60


def test_tei_full_result_summary():
    result = TEIFullResult(
        baseline_eval=_make_eval_result(0.60, False),
        final_eval=_make_eval_result(0.85, True),
        total_duration_ms=5000,
        total_cost_usd=0.05,
    )
    summary = result.summary()
    assert "0.600" in summary
    assert "0.850" in summary
    assert "+0.250" in summary
