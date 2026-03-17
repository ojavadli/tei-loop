"""
Base judge class for all TEI evaluation dimensions.

Each dimension judge:
1. Takes a Trace (agent input + output + optional steps)
2. Extracts verifiable assertions from the output
3. Checks each assertion against evidence
4. Returns a DimensionScore with score, assertions, and reasoning
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import (
    Assertion,
    AssertionVerdict,
    Dimension,
    DimensionScore,
    Trace,
)
from ..llm_provider import BaseLLMProvider


class BaseJudge(ABC):
    """Abstract base for a TEI evaluation dimension judge."""

    dimension: Dimension

    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @abstractmethod
    def build_user_prompt(self, trace: Trace) -> str:
        ...

    async def evaluate(
        self,
        trace: Trace,
        llm: BaseLLMProvider,
        threshold: float = 0.7,
    ) -> DimensionScore:
        system = self.system_prompt()
        user = self.build_user_prompt(trace)

        try:
            result = await llm.generate_json(system, user)
        except Exception as e:
            return DimensionScore(
                dimension=self.dimension,
                score=0.0,
                passed=False,
                threshold=threshold,
                reasoning=f"Evaluation failed: {e}",
                failure_summary=str(e),
            )

        score = float(result.get("score", 0))
        score = max(0.0, min(0.97, score))

        assertions = []
        for a in result.get("assertions", []):
            verdict_str = a.get("verdict", "fail").lower()
            if verdict_str == "pass":
                verdict = AssertionVerdict.PASS
            elif verdict_str == "partial":
                verdict = AssertionVerdict.PARTIAL
            else:
                verdict = AssertionVerdict.FAIL

            assertions.append(Assertion(
                claim=a.get("claim", ""),
                evidence=a.get("evidence", ""),
                verdict=verdict,
                dimension=self.dimension,
                explanation=a.get("explanation", ""),
            ))

        passed_count = sum(1 for a in assertions if a.verdict == AssertionVerdict.PASS)
        total_count = len(assertions)

        return DimensionScore(
            dimension=self.dimension,
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            assertions=assertions,
            reasoning=result.get("reasoning", ""),
            failure_summary=result.get("failure_summary") if score < threshold else None,
        )
