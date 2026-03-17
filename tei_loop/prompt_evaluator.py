from .models import MetricFormula, MetricResult, Trace
from .llm_provider import BaseLLMProvider
import json
import re


def _evaluate_metric_by_code(
    metric: MetricFormula, agent_input: str, agent_output: str
) -> tuple[float, str]:
    """Try to evaluate a metric using code-based methods. Returns (score, detail)."""
    inp = agent_input or ""
    out = agent_output or ""
    if isinstance(inp, (dict, list)):
        inp = json.dumps(inp, indent=2)
    if isinstance(out, (dict, list)):
        out = json.dumps(out, indent=2)
    inp = str(inp)
    out = str(out)

    name_lower = metric.name.lower()
    formula_lower = metric.formula.lower()

    if "coverage" in name_lower or "coverage" in formula_lower or "count(" in formula_lower and "/" in formula_lower:
        items = re.findall(r"\b[A-Za-z][A-Za-z0-9_\s]{2,40}\b", inp)
        items = [x.strip() for x in items if len(x.strip()) > 2][:50]
        if not items:
            items = re.split(r"[\s,;:\n]+", inp)
            items = [x for x in items if len(x) > 2][:50]
        found = sum(1 for item in items if item.lower() in out.lower())
        total = max(len(items), 1)
        score = 100.0 * found / total
        return (score, f"{found}/{total}")

    if "presence" in name_lower or "keyword" in name_lower or "presence" in formula_lower:
        keywords = re.findall(r"\b[A-Za-z][A-Za-z0-9_]{2,30}\b", inp)
        keywords = list(dict.fromkeys([x.strip() for x in keywords if len(x.strip()) > 2]))[:30]
        if not keywords:
            keywords = re.split(r"[\s,;:\n]+", inp)
            keywords = [x for x in keywords if len(x) > 2][:30]
        found = sum(1 for kw in keywords if kw.lower() in out.lower())
        total = max(len(keywords), 1)
        score = 100.0 * found / total
        return (score, f"{found}/{total}")

    if "ratio" in name_lower or "length" in name_lower or "length" in formula_lower:
        len_in = max(len(inp), 1)
        len_out = len(out)
        ratio = len_out / len_in
        if ratio < 0.5:
            score = 50.0 * ratio / 0.5
        elif ratio > 2.0:
            score = max(0, 100 - 50 * (ratio - 2.0))
        else:
            score = 50 + 50 * (1.0 - abs(ratio - 1.0))
        score = max(0.0, min(100.0, score))
        return (score, f"{len_out}/{len_in}")

    if "compliance" in name_lower or "format" in name_lower or "structure" in name_lower or "json" in formula_lower:
        checks = 0
        passed = 0
        if "json" in formula_lower or "json" in name_lower:
            checks += 1
            try:
                json.loads(out)
                passed += 1
            except (json.JSONDecodeError, TypeError):
                pass
        if "section" in formula_lower or "header" in formula_lower:
            checks += 1
            headers = len(re.findall(r"^#{1,6}\s+.+$|^[A-Z][a-z]+:.*$", out, re.MULTILINE))
            passed += 1 if headers > 0 else 0
        if checks == 0:
            checks = 1
            passed = 1 if out.strip() else 0
        score = 100.0 * passed / max(checks, 1)
        return (score, f"{passed}/{checks}")

    if "entity" in name_lower or "retention" in name_lower:
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b[A-Z]{2,}\b", inp)
        entities = list(dict.fromkeys([x.strip() for x in entities if len(x.strip()) > 1]))[:40]
        if not entities:
            entities = re.findall(r"\b[A-Za-z][A-Za-z0-9_]{3,25}\b", inp)
            entities = list(dict.fromkeys(entities))[:40]
        found = sum(1 for e in entities if e in out or e.lower() in out.lower())
        total = max(len(entities), 1)
        score = 100.0 * found / total
        return (score, f"{found}/{total}")

    if "completeness" in name_lower or "question" in name_lower or "answered" in formula_lower:
        questions = re.findall(r"[?].+|[Qq]uestion\s*\d*[.:]?\s*.+", inp)
        if not questions:
            questions = re.split(r"\n+", inp)
            questions = [q.strip() for q in questions if "?" in q or len(q) > 20][:20]
        total = max(len(questions), 1)
        out_sentences = re.split(r"[.!?\n]+", out)
        out_text = " ".join(out_sentences).lower()
        answered = sum(1 for q in questions if any(w in out_text for w in q.split()[:5] if len(w) > 2))
        score = 100.0 * answered / total
        return (score, f"{answered}/{total}")

    return (0.0, "unable to parse")


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
        agent_input = trace.agent_input
        agent_output = trace.agent_output
        if isinstance(agent_input, (dict, list)):
            agent_input = json.dumps(agent_input, indent=2)
        elif agent_input is None:
            agent_input = ""
        if isinstance(agent_output, (dict, list)):
            agent_output = json.dumps(agent_output, indent=2)
        elif agent_output is None:
            agent_output = ""
        inp_str = str(agent_input)
        out_str = str(agent_output)

        for metric in metrics:
            use_code = metric.measurement_method in ("code_based", "hybrid")
            score = 0.0
            detail = ""
            used_llm = False

            if use_code:
                score, detail = _evaluate_metric_by_code(metric, inp_str, out_str)
                if detail == "unable to parse" and metric.measurement_method == "code_based":
                    use_code = False

            if not use_code or (metric.measurement_method == "hybrid" and detail == "unable to parse"):
                system = self._build_system_prompt()
                user = self._build_user_prompt(trace, metric)
                raw = await self.eval_llm.generate(system_prompt=system, user_prompt=user)
                parsed = self._parse_llm_response(raw)
                score = float(parsed.get("score", 0))
                score = max(0.0, min(100.0, score))
                detail = str(parsed.get("detail", ""))
                used_llm = True

            results.append(
                MetricResult(
                    metric=metric,
                    score=score,
                    detail=detail,
                    raw_data={"used_llm_fallback": used_llm},
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
