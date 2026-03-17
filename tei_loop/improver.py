"""
TEI Improver Module.

The novel contribution: maps evaluation failures to targeted fix strategies.
Unlike prompt optimizers that blindly search, TEI diagnoses the specific failure
dimension and applies the right fix type.

Four fix strategies (one per dimension):
1. Target Re-anchor: re-prompt with original objective when agent drifts
2. Reasoning Regenerate: rebuild the plan when reasoning is flawed
3. Execution Correct: fix tool calls, parameters, API handling
4. Output Repair: patch factual errors, fill gaps, resolve contradictions
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .models import (
    Dimension,
    EvalResult,
    Failure,
    Fix,
    FixStrategyType,
    Severity,
    TEIConfig,
    Trace,
)
from .llm_provider import BaseLLMProvider


# ---------------------------------------------------------------------------
# Failure classifier (Step 41)
# ---------------------------------------------------------------------------

def classify_failures(eval_result: EvalResult) -> list[Failure]:
    """Extract and rank failures from an EvalResult.

    Returns failures sorted by severity (critical first), each mapped
    to the appropriate fix strategy based on its dimension.
    """
    dimension_to_strategy = {
        Dimension.TARGET_ALIGNMENT: FixStrategyType.TARGET_REANCHOR,
        Dimension.REASONING_SOUNDNESS: FixStrategyType.REASONING_REGENERATE,
        Dimension.EXECUTION_ACCURACY: FixStrategyType.EXECUTION_CORRECT,
        Dimension.OUTPUT_INTEGRITY: FixStrategyType.OUTPUT_REPAIR,
    }

    failures: list[Failure] = []
    for dim_score in eval_result.get_failures():
        gap = dim_score.threshold - dim_score.score
        if gap >= 0.4:
            severity = Severity.CRITICAL
        elif gap >= 0.2:
            severity = Severity.MAJOR
        else:
            severity = Severity.MINOR

        failures.append(Failure(
            dimension=dim_score.dimension,
            severity=severity,
            description=dim_score.failure_summary or dim_score.reasoning,
            evidence="; ".join(
                a.explanation for a in dim_score.assertions
                if a.verdict.value != "pass"
            )[:1000],
            suggested_strategy=dimension_to_strategy.get(
                dim_score.dimension, FixStrategyType.OUTPUT_REPAIR
            ),
        ))

    severity_order = {Severity.CRITICAL: 0, Severity.MAJOR: 1, Severity.MINOR: 2}
    failures.sort(key=lambda f: severity_order.get(f.severity, 3))
    return failures


# ---------------------------------------------------------------------------
# Fix strategy prompts (Steps 42-45)
# ---------------------------------------------------------------------------

STRATEGY_PROMPTS: dict[FixStrategyType, str] = {
    FixStrategyType.TARGET_REANCHOR: """You are a TEI improvement engine.

The agent DRIFTED from its target. The evaluation found that the agent's output
does not align with what the user actually asked for.

Your job: generate a corrected instruction that re-anchors the agent to the
original objective. This instruction will be prepended to the user's query
on the next agent run.

ORIGINAL USER QUERY:
{query}

AGENT OUTPUT (that drifted):
{output}

FAILURE DIAGNOSIS:
{failure_description}

EVIDENCE:
{evidence}

Return JSON:
{{
  "reanchored_instruction": "A clear, specific instruction that re-focuses the agent on the actual objective. Include what was missed or misinterpreted.",
  "rationale": "Why the agent drifted and how this instruction fixes it"
}}""",

    FixStrategyType.REASONING_REGENERATE: """You are a TEI improvement engine.

The agent's REASONING was flawed. The evaluation found logical errors,
contradictions, or unsupported conclusions in the agent's thought process.

Your job: generate guidance that helps the agent reason more soundly on retry.

ORIGINAL USER QUERY:
{query}

AGENT OUTPUT (with flawed reasoning):
{output}

FAILURE DIAGNOSIS:
{failure_description}

EVIDENCE:
{evidence}

Return JSON:
{{
  "reasoning_guidance": "Specific guidance for the agent to avoid the reasoning errors found. Reference the actual errors and how to fix them.",
  "constraints": ["List of logical constraints the agent must follow"],
  "rationale": "What reasoning errors were found and how this guidance addresses them"
}}""",

    FixStrategyType.EXECUTION_CORRECT: """You are a TEI improvement engine.

The agent's EXECUTION was incorrect. The evaluation found wrong tool calls,
incorrect parameters, mishandled responses, or skipped steps.

Your job: generate corrections for the execution errors.

ORIGINAL USER QUERY:
{query}

AGENT OUTPUT (with execution errors):
{output}

FAILURE DIAGNOSIS:
{failure_description}

EVIDENCE:
{evidence}

Return JSON:
{{
  "execution_corrections": "Specific corrections for the tool calls, parameters, or steps that were wrong. Be precise about what was wrong and what should be done instead.",
  "guardrails": ["List of constraints to prevent the same execution errors"],
  "rationale": "What execution errors occurred and how these corrections fix them"
}}""",

    FixStrategyType.OUTPUT_REPAIR: """You are a TEI improvement engine.

The agent's OUTPUT has integrity issues. The evaluation found factual errors,
missing information, inconsistencies, or formatting problems.

Your job: generate specific instructions to repair the output.

ORIGINAL USER QUERY:
{query}

AGENT OUTPUT (with integrity issues):
{output}

FAILURE DIAGNOSIS:
{failure_description}

EVIDENCE:
{evidence}

Return JSON:
{{
  "repair_instructions": "Specific instructions for fixing the output issues. Reference what is wrong and what the correct content should be.",
  "missing_elements": ["List of information that must be included"],
  "corrections": ["List of factual or consistency corrections needed"],
  "rationale": "What integrity issues were found and how these repairs address them"
}}""",
}


# ---------------------------------------------------------------------------
# Improvement engine (Steps 46-52)
# ---------------------------------------------------------------------------

class TEIImprover:
    """Generates targeted fixes based on evaluation failures."""

    def __init__(self, config: TEIConfig, improve_llm: BaseLLMProvider):
        self.config = config
        self.improve_llm = improve_llm

    async def generate_fixes(
        self,
        trace: Trace,
        failures: list[Failure],
        max_fixes: int = 3,
    ) -> list[Fix]:
        """Generate fixes for the top failures. Prioritizes by severity."""
        fixes: list[Fix] = []

        for failure in failures[:max_fixes]:
            fix = await self._generate_single_fix(trace, failure)
            if fix:
                fixes.append(fix)

        return fixes

    async def _generate_single_fix(
        self,
        trace: Trace,
        failure: Failure,
    ) -> Optional[Fix]:
        """Generate a single fix for a single failure."""
        prompt_template = STRATEGY_PROMPTS.get(failure.suggested_strategy)
        if not prompt_template:
            return None

        user_prompt = prompt_template.format(
            query=str(trace.agent_input or "")[:3000],
            output=str(trace.agent_output or "")[:4000],
            failure_description=failure.description[:1000],
            evidence=failure.evidence[:1000],
        )

        system_prompt = (
            "You are a TEI improvement engine. Generate precise, actionable fixes. "
            "Return valid JSON only."
        )

        try:
            result = await self.improve_llm.generate_json(system_prompt, user_prompt)
        except Exception as e:
            return Fix(
                strategy=failure.suggested_strategy,
                failure_id=failure.failure_id,
                rationale=f"Fix generation failed: {e}",
            )

        def _to_str(val: Any) -> str:
            if isinstance(val, str):
                return val
            if isinstance(val, list):
                return "; ".join(str(v) for v in val)
            return str(val) if val else ""

        def _join_list(val: Any) -> list:
            if isinstance(val, list):
                return [str(v) for v in val]
            if isinstance(val, str) and val:
                return [val]
            return []

        replacement = ""
        if failure.suggested_strategy == FixStrategyType.TARGET_REANCHOR:
            replacement = _to_str(result.get("reanchored_instruction", ""))
        elif failure.suggested_strategy == FixStrategyType.REASONING_REGENERATE:
            guidance = _to_str(result.get("reasoning_guidance", ""))
            constraints = _join_list(result.get("constraints", []))
            replacement = guidance
            if constraints:
                replacement += "\nConstraints: " + "; ".join(constraints)
        elif failure.suggested_strategy == FixStrategyType.EXECUTION_CORRECT:
            corrections = _to_str(result.get("execution_corrections", ""))
            guardrails = _join_list(result.get("guardrails", []))
            replacement = corrections
            if guardrails:
                replacement += "\nGuardrails: " + "; ".join(guardrails)
        elif failure.suggested_strategy == FixStrategyType.OUTPUT_REPAIR:
            repairs = _to_str(result.get("repair_instructions", ""))
            missing = _join_list(result.get("missing_elements", []))
            corrections = _join_list(result.get("corrections", []))
            replacement = repairs
            if missing:
                replacement += "\nMissing: " + "; ".join(missing)
            if corrections:
                replacement += "\nCorrections: " + "; ".join(corrections)

        return Fix(
            strategy=failure.suggested_strategy,
            failure_id=failure.failure_id,
            original_content=str(trace.agent_input or ""),
            replacement_content=replacement,
            rationale=result.get("rationale", ""),
            metadata=result,
        )

    def build_improved_query(
        self,
        original_query: Any,
        fixes: list[Fix],
    ) -> str:
        """Combine original query with all fixes into an improved prompt for re-run."""
        query_str = str(original_query)
        improvements: list[str] = []

        for fix in fixes:
            if fix.replacement_content:
                label = fix.strategy.value.replace("_", " ").title()
                improvements.append(
                    f"[TEI {label}]: {fix.replacement_content}"
                )

        if not improvements:
            return query_str

        improved = (
            "IMPORTANT CONTEXT FROM PRIOR EVALUATION:\n"
            + "\n\n".join(improvements)
            + "\n\nORIGINAL QUERY:\n"
            + query_str
        )
        return improved


# ---------------------------------------------------------------------------
# Convergence checker (Step 49)
# ---------------------------------------------------------------------------

def check_convergence(
    scores: list[float],
    threshold: float = 0.02,
    window: int = 3,
) -> bool:
    """Check if scores have converged (improvement plateaued).

    Returns True if the last `window` scores differ by less than `threshold`.
    """
    if len(scores) < window:
        return False
    recent = scores[-window:]
    return (max(recent) - min(recent)) < threshold
