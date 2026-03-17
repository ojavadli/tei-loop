"""
TEI Evaluator.

Orchestrates all 4 dimension judges, runs them in parallel (async),
aggregates scores, and produces a complete EvalResult.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from .models import (
    Dimension,
    DimensionConfig,
    DimensionScore,
    EvalResult,
    TEIConfig,
    Trace,
)
from .dimensions import ALL_JUDGES
from .dimensions.base import BaseJudge
from .llm_provider import BaseLLMProvider


class TEIEvaluator:
    """Runs all TEI evaluation dimensions and produces an EvalResult."""

    def __init__(self, config: TEIConfig, eval_llm: BaseLLMProvider):
        self.config = config
        self.eval_llm = eval_llm
        self._judges: list[BaseJudge] = [cls() for cls in ALL_JUDGES]

    async def evaluate(self, trace: Trace) -> EvalResult:
        """Run all enabled dimension judges on a trace. Returns aggregated EvalResult."""
        start = time.time()

        enabled_judges = [
            j for j in self._judges
            if self.config.dimensions.get(j.dimension, DimensionConfig()).enabled
        ]

        if self.config.parallel_eval:
            tasks = [
                self._run_judge(j, trace)
                for j in enabled_judges
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for j in enabled_judges:
                try:
                    r = await self._run_judge(j, trace)
                    results.append(r)
                except Exception as e:
                    results.append(e)

        dimension_scores: dict[Dimension, DimensionScore] = {}
        for j, result in zip(enabled_judges, results):
            if isinstance(result, Exception):
                dimension_scores[j.dimension] = DimensionScore(
                    dimension=j.dimension,
                    score=0.0,
                    passed=False,
                    threshold=self.config.dimensions.get(
                        j.dimension, DimensionConfig()
                    ).threshold,
                    reasoning=f"Judge failed: {result}",
                    failure_summary=str(result),
                )
            else:
                dimension_scores[j.dimension] = result

        total_assertions = sum(len(ds.assertions) for ds in dimension_scores.values())
        passed_assertions = sum(
            sum(1 for a in ds.assertions if a.verdict.value == "pass")
            for ds in dimension_scores.values()
        )

        weights = {
            dim: self.config.dimensions.get(dim, DimensionConfig()).weight
            for dim in dimension_scores
        }
        total_weight = sum(weights.values()) or 1.0
        aggregate = sum(
            dimension_scores[dim].score * weights[dim]
            for dim in dimension_scores
        ) / total_weight

        all_passed = all(ds.passed for ds in dimension_scores.values())

        elapsed_ms = (time.time() - start) * 1000
        cost = self.eval_llm.get_cost("openai")

        return EvalResult(
            trace_id=trace.trace_id,
            dimension_scores=dimension_scores,
            aggregate_score=round(aggregate, 4),
            all_passed=all_passed,
            total_assertions=total_assertions,
            passed_assertions=passed_assertions,
            failed_assertions=total_assertions - passed_assertions,
            eval_duration_ms=elapsed_ms,
            eval_cost_usd=cost,
        )

    async def _run_judge(self, judge: BaseJudge, trace: Trace) -> DimensionScore:
        threshold = self.config.dimensions.get(
            judge.dimension, DimensionConfig()
        ).threshold
        return await judge.evaluate(trace, self.eval_llm, threshold)

    async def evaluate_single(
        self, trace: Trace, dimension: Dimension
    ) -> DimensionScore:
        """Evaluate a single dimension only."""
        judge = next((j for j in self._judges if j.dimension == dimension), None)
        if not judge:
            raise ValueError(f"No judge for dimension: {dimension}")
        threshold = self.config.dimensions.get(dimension, DimensionConfig()).threshold
        return await judge.evaluate(trace, self.eval_llm, threshold)
