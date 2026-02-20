from .models import MetricFormula, MetricResult, Trace
from .llm_provider import BaseLLMProvider
import json
import re


class PromptEvaluator:
    def __init__(self, eval_llm: BaseLLMProvider):
        self.eval_llm = eval_llm

    def _build_system_prompt(self) -> str:
        return (
            "You are a prompt efficiency evaluator. Score how well the agent's output meets a specific quality metric. "
            "Be strict and objective. Use the metric definition and formula to guide your scoring. "
            "Return valid JSON only."
        )

    def _build_user_prompt(self, trace: Trace, metric: MetricFormula) -> str:
        agent_input = trace.agent_input
        agent_output = trace.agent_output
        if isinstance(agent_input, (dict, list)):
            agent_input = json.dumps(agent_input, indent=2)
        elif agent_input is None:
            agent_input = "(none)"
        if isinstance(agent_output, (dict, list)):
            agent_output = json.dumps(agent_output, indent=2)
        elif agent_output is None:
            agent_output = "(none)"

        return f"""## Metric
**Name:** {metric.name}
**Description:** {metric.description}
**Formula:** {metric.formula}

## Agent Input
```
{agent_input}
```

## Agent Output
```
{agent_output}
```

## Task
Score how well the agent's output satisfies this metric. Provide:
1. **score**: integer 0-100 (0 = complete failure, 100 = perfect)
2. **detail**: brief quantitative summary (e.g. "8/12 topics found", "3/5 requirements met")
3. **reasoning**: short explanation of your scoring

Respond with valid JSON only:
{{"score": <0-100>, "detail": "<brief summary>", "reasoning": "<explanation>"}}"""

    def _parse_llm_response(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[^{}]*\"score\"[^{}]*\"detail\"[^{}]*\"reasoning\"[^{}]*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    async def evaluate_prompt(
        self,
        trace: Trace,
        metrics: list[MetricFormula],
    ) -> list[MetricResult]:
        results: list[MetricResult] = []
        system = self._build_system_prompt()
        for metric in metrics:
            user = self._build_user_prompt(trace, metric)
            raw = await self.eval_llm.generate(system_prompt=system, user_prompt=user)
            parsed = self._parse_llm_response(raw)
            score = float(parsed.get("score", 0))
            score = max(0.0, min(100.0, score))
            detail = str(parsed.get("detail", ""))
            reasoning = str(parsed.get("reasoning", ""))
            results.append(
                MetricResult(
                    metric=metric,
                    score=score,
                    detail=detail,
                    raw_data={"reasoning": reasoning, "raw_response": parsed},
                )
            )
        return results

    async def evaluate_batch(
        self,
        traces: list[Trace],
        metrics: list[MetricFormula],
    ) -> list[MetricResult]:
        all_results: list[list[MetricResult]] = []
        for trace in traces:
            results = await self.evaluate_prompt(trace, metrics)
            all_results.append(results)
        if not all_results:
            return []
        averaged: list[MetricResult] = []
        for i, metric in enumerate(metrics):
            scores = [r[i].score for r in all_results]
            details = [r[i].detail for r in all_results]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            combined_detail = "; ".join(details) if len(details) <= 3 else f"avg {avg_score:.1f} across {len(traces)} traces"
            averaged.append(
                MetricResult(
                    metric=metric,
                    score=avg_score,
                    detail=combined_detail,
                    raw_data={"trace_count": len(traces), "scores": scores},
                )
            )
        return averaged

    def compute_composite(
        self,
        results: list[MetricResult],
    ) -> float:
        if not results:
            return 0.0
        total_weight = sum(r.metric.weight for r in results)
        if total_weight <= 0:
            return sum(r.score for r in results) / len(results)
        weighted_sum = sum(r.score * r.metric.weight for r in results)
        return weighted_sum / total_weight
