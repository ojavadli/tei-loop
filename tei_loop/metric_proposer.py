"""
Metric proposer for task-specific objective metrics.

Proposes measurable metrics for prompt output quality.
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
        system_prompt = """You are an expert at designing objective metrics for evaluating AI agent outputs.
Your task is to propose task-specific metrics that measure how well a prompt's OUTPUT achieves the agent's goal.
Each metric must be measurable: either by code (string matching, counting, regex) or by an LLM judge.
Be specific and actionable. Avoid vague metrics like "quality" or "helpfulness"."""

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

Propose 3-5 objective metrics that measure how well the prompt's OUTPUT achieves the task goal.
Each metric should be measurable - either by code (string matching, counting, regex) or by LLM judge.
Include an explicit formula for each metric.

Respond with valid JSON only:
{{
  "metrics": [
    {{
      "name": "Metric Name",
      "description": "What this metric measures and why it matters",
      "formula": "Human-readable formula, e.g. 'count(required_topics_in_output) / total_required_topics'",
      "measurement_method": "code_based" or "llm_judge" or "hybrid"
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
            formula = m.get("formula", "")
            method = m.get("measurement_method", "llm_judge")
            if method not in ("code_based", "llm_judge", "hybrid"):
                method = "llm_judge"
            formulas.append(
                MetricFormula(
                    name=name,
                    description=description,
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
