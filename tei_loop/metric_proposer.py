"""
Metric proposer for task-specific objective metrics.

Proposes measurable metrics for prompt output quality, for prompt output quality.
"""

from __future__ import annotations

import json

from .llm_provider import BaseLLMProvider
from .models import Checkpoint, Dimension, MetricFormula


class MetricProposer:
    def __init__(self, improve_llm: BaseLLMProvider):
        self.improve_llm = improve_llm

    async def propose_metrics(
        self,
        checkpoints: list[Checkpoint],
        agent_source: str,
        prompt_text: str,
    ) -> list[MetricFormula]:
        system_prompt = """You are an expert at designing meaningful metrics for evaluating AI agent outputs.
Your task is to propose task-specific metrics that measure OUTPUT QUALITY from the END USER's perspective.

Focus on what matters to the person receiving the agent's response:
- Did the response actually solve the user's problem?
- Was empathy and tone appropriate?
- Were concrete, actionable next steps provided?
- Was the response complete (not leaving the user hanging)?
- Was information accurate and grounded in real data?

Metrics can be measured by LLM judgment (preferred for quality) or by code (for structural checks).
Mix both types: at least 2 LLM-judged metrics for quality, and up to 2 code-based for structure.
Be specific and actionable."""

        checkpoint_summary = ""
        if checkpoints:
            checkpoint_summary = "\n\nCheckpoints (key decision points in the agent):\n"
            for cp in checkpoints[:20]:
                checkpoint_summary += f"- {cp.checkpoint_type} at {cp.file_path}:{cp.line_number} ({cp.dimension.value})\n"

        user_prompt = f"""Analyze this agent's prompts and the task it performs.

AGENT SOURCE CODE:
```
{agent_source}
```

PROMPT TEXT (the prompt being evaluated):
```
{prompt_text}
```
{checkpoint_summary}

Propose 5 metrics that measure how well the agent's OUTPUT serves the end user.
Focus on outcome quality, not internal tool mechanics.

IMPORTANT: Derive metrics from the SPECIFIC agent and task above. Do NOT use generic templates.
Look at the actual prompt text, the domain the agent operates in, and what specific quality
issues could arise. For example:
- If the agent handles orders, propose metrics about order-data accuracy and completeness
- If the agent summarizes, propose metrics about coverage, conciseness, factual grounding
- If the agent classifies, propose metrics about label accuracy and confidence calibration

Each metric MUST have a clear pass/fail criterion or scoring rubric. Avoid vague metrics
like "quality" or "helpfulness" — be specific about what is measured and how.

Each metric should use "llm_judge" (LLM evaluates quality) or "code_based" (string matching / presence checks).
Use "llm_judge" for subjective quality metrics, "code_based" for structural/factual checks.
Include at least 2 "code_based" metrics and at least 2 "llm_judge" metrics.
Include an explicit formula or evaluation criteria for each metric.

Respond with valid JSON only:
{{
  "metrics": [
    {{
      "name": "Metric Name",
      "description": "What this metric measures and why it matters for the end user",
      "rationale": "Why you are proposing this metric — what specific weakness or pattern in the agent's output led you to suggest it",
      "formula": "Evaluation criteria or formula",
      "measurement_method": "llm_judge or code_based"
    }}
  ]
}}"""

        try:
            result = await self.improve_llm.generate_json(system_prompt, user_prompt)
        except json.JSONDecodeError:
            text = await self.improve_llm.generate(system_prompt, user_prompt)
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            result = json.loads(text)

        metrics_data = result.get("metrics", [])
        if not isinstance(metrics_data, list):
            return []

        formulas: list[MetricFormula] = []
        n = len(metrics_data)
        default_weight = 1.0 / max(n, 1) if n > 0 else 0.25

        for i, m in enumerate(metrics_data[:5]):
            if not isinstance(m, dict):
                continue
            name = m.get("name", f"Metric_{i+1}")
            description = m.get("description", "")
            rationale = m.get("rationale", "")
            formula = m.get("formula", "")
            method = m.get("measurement_method", "code_based")
            if method not in ("code_based", "llm_judge", "hybrid"):
                method = "code_based"
            formulas.append(
                MetricFormula(
                    name=name,
                    description=description,
                    rationale=rationale,
                    formula=formula,
                    measurement_method=method,
                    weight=default_weight,
                )
            )

        return formulas

    async def propose_composite_weights(
        self,
        metrics: list[MetricFormula],
    ) -> dict[str, float]:
        if not metrics:
            return {}

        equal_weight = 1.0 / len(metrics)
        weights = {m.name: equal_weight for m in metrics}

        if len(metrics) <= 1:
            return weights

        system_prompt = """You suggest weights for a composite score. Weights must sum to 1.0.
Consider which metrics are most critical for the task. Respond with valid JSON only."""

        metric_list = "\n".join(
            f"- {m.name}: {m.description}" for m in metrics
        )
        user_prompt = f"""Given these metrics for evaluating an agent's output:

{metric_list}

Suggest weights (0.0 to 1.0) for each metric. Weights must sum to exactly 1.0.
More important metrics should have higher weights.

Respond with valid JSON only:
{{
  "weights": {{
    "Metric Name 1": 0.35,
    "Metric Name 2": 0.30,
    ...
  }}
}}"""

        try:
            result = await self.improve_llm.generate_json(system_prompt, user_prompt)
        except (json.JSONDecodeError, Exception):
            return weights

        raw = result.get("weights", {})
        if not isinstance(raw, dict):
            return weights

        total = sum(v for v in raw.values() if isinstance(v, (int, float)))
        if total <= 0:
            return weights

        for name, w in raw.items():
            if name in weights and isinstance(w, (int, float)) and w >= 0:
                weights[name] = w / total

        normalized_total = sum(weights.values())
        if abs(normalized_total - 1.0) > 0.01:
            for name in weights:
                weights[name] /= normalized_total

        return weights


def format_composite_formula(metrics: list[MetricFormula]) -> str:
    if not metrics:
        return "Composite = (no metrics)"
    parts = [f"{m.weight:.2f} * {m.name}" for m in metrics]
    return "Composite = " + " + ".join(parts)
