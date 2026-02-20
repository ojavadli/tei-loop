from .models import MetricFormula, MetricResult, ParetoCandidate, OptimizationResult, Trace
from .llm_provider import BaseLLMProvider
from .pareto import update_pareto_front, sample_from_front, sample_pair_from_front, compute_composite, select_best
from .prompt_evaluator import PromptEvaluator
from .tracer import run_and_trace
from .prompt_improver import extract_prompts, create_patched_agent
import random
import json
from typing import Any, Callable, Optional


class PromptOptimizer:
    def __init__(
        self,
        improve_llm: BaseLLMProvider,
        eval_llm: BaseLLMProvider,
        metrics: list[MetricFormula],
        agent_fn: Callable,
        agent_file: Optional[str] = None,
    ):
        self.improve_llm = improve_llm
        self.eval_llm = eval_llm
        self.metrics = metrics
        self.agent_fn = agent_fn
        self.agent_file = agent_file
        self.prompt_evaluator = PromptEvaluator(eval_llm)
        self.rng = random.Random(42)

    async def optimize(
        self,
        original_prompt: str,
        test_queries: list[Any],
        num_iterations: int = 30,
        verbose: bool = True,
    ) -> OptimizationResult:
        original_prompts = extract_prompts(self.agent_fn, self.agent_file)
        if not original_prompts:
            original_prompts = {"user_prompt_template": original_prompt}
        elif "user_prompt_template" not in original_prompts and "system_prompt" not in original_prompts:
            original_prompts["user_prompt_template"] = original_prompt

        patched = self._create_patched_agent(original_prompt)
        traces, metric_results, composite = await self._run_and_evaluate(patched, test_queries, self.metrics)
        metric_scores = {r.metric.name: r.score / 100.0 for r in metric_results}
        weights = self._metric_weights()
        comp_pct = compute_composite(metric_scores, weights)

        p0 = ParetoCandidate(
            iteration=0,
            prompt_text=original_prompt,
            metric_scores={r.metric.name: r.score for r in metric_results},
            composite_score=comp_pct,
            strategy="baseline",
            reflection="",
        )
        front = [p0]
        metric_history = [dict(p0.metric_scores)]
        baseline_scores = dict(p0.metric_scores)

        for i in range(1, num_iterations + 1):
            use_merge = len(front) >= 2 and self.rng.random() < 0.30
            if use_merge:
                ca, cb = sample_pair_from_front(front, self.rng)
                new_prompt = await self._system_aware_merge(ca, cb)
                strategy_name = "merge"
                parent_a, parent_b = ca, cb
            else:
                parent = sample_from_front(front, self.rng)
                parent_patched = self._create_patched_agent(parent.prompt_text)
                sample_q = self.rng.choice(test_queries)
                parent_trace = await run_and_trace(parent_patched, sample_q)
                parent_results = [MetricResult(metric=m, score=parent.metric_scores.get(m.name, 0), detail="") for m in self.metrics]
                new_prompt = await self._reflective_mutation(parent, parent_trace, parent_results)
                strategy_name = "mutation"
                parent_a, parent_b = parent, None

            patched = self._create_patched_agent(new_prompt)
            batch = self.rng.sample(test_queries, min(3, len(test_queries)))
            batch_traces, batch_results, _ = await self._run_and_evaluate(patched, batch, self.metrics)
            metric_scores_new = {r.metric.name: r.score / 100.0 for r in batch_results}
            comp_new = compute_composite(metric_scores_new, weights)
            metric_scores_raw = {r.metric.name: r.score for r in batch_results}

            new_candidate = ParetoCandidate(
                iteration=i,
                prompt_text=new_prompt,
                metric_scores=metric_scores_raw,
                composite_score=comp_new,
                parent_ids=[parent_a.candidate_id] + ([parent_b.candidate_id] if parent_b else []),
                strategy=strategy_name,
                reflection="",
            )
            old_len = len(front)
            front = update_pareto_front(front, new_candidate)
            added = len(front) > old_len
            metric_history.append(metric_scores_raw)

            if verbose:
                abbrevs = self._metric_abbrevs(metric_scores_raw, baseline_scores)
                delta_str = ", ".join(abbrevs) if abbrevs else ""
                add_str = f" new Pareto candidate ({delta_str})" if added else ""
                ref_preview = ""
                if strategy_name == "mutation":
                    ref_preview = new_prompt[:80].replace("\n", " ") + "..." if len(new_prompt) > 80 else new_prompt[:80]
                else:
                    ref_preview = f"merged from P{parent_a.iteration} + P{parent_b.iteration}"
                print(f"  Iter {i:2}/{num_iterations} | Comp: {comp_new:.1f}% | Pool: {len(front)}{add_str}")
                print(f"    {strategy_name.capitalize()} from P{parent_a.iteration}. {ref_preview}")

        best = select_best(front)
        return OptimizationResult(
            total_iterations=num_iterations,
            pareto_front=front,
            best_candidate=best,
            metric_history=metric_history,
            baseline_scores=baseline_scores,
            final_scores=best.metric_scores,
        )

    def _metric_weights(self) -> dict[str, float]:
        total = sum(m.weight for m in self.metrics)
        if total <= 0:
            return {m.name: 1.0 / len(self.metrics) for m in self.metrics}
        return {m.name: m.weight / total for m in self.metrics}

    def _metric_abbrevs(self, new_scores: dict[str, float], base_scores: dict[str, float]) -> list[str]:
        out = []
        for m in self.metrics:
            n = new_scores.get(m.name, 0)
            b = base_scores.get(m.name, 0)
            delta = round(n - b)
            abbr = "".join(w[0].upper() for w in m.name.split()[:2])[:2] or m.name[:2]
            out.append(f"{abbr}{delta:+d}")
        return out

    async def _reflective_mutation(
        self,
        candidate: ParetoCandidate,
        trace: Trace,
        metric_results: list[MetricResult],
    ) -> str:
        inp = trace.agent_input
        out = trace.agent_output
        if isinstance(inp, (dict, list)):
            inp = json.dumps(inp, indent=2)
        if isinstance(out, (dict, list)):
            out = json.dumps(out, indent=2)
        inp = str(inp) if inp is not None else "(none)"
        out = str(out) if out is not None else "(none)"

        lines = []
        for r in metric_results:
            lines.append(f"- {r.metric.name}: {r.score:.1f} - {r.detail or r.raw_data.get('reasoning', '')}")

        prompt = f"""You are a prompt engineer. An agent uses the following prompt and produced the trace below.

CURRENT PROMPT:
```
{candidate.prompt_text}
```

AGENT TRACE (input -> output):
Input:
```
{inp[:3000]}
```

Output:
```
{out[:3000]}
```

METRIC SCORES:
{chr(10).join(lines)}

Reflect on why these metrics scored as they did. Propose a specific improved version of the prompt that addresses the weakest metrics. Return only the new prompt text, no JSON or extra commentary."""

        raw = await self.improve_llm.generate(
            system_prompt="You are an expert prompt engineer. Return only the improved prompt text.",
            user_prompt=prompt,
        )
        return raw.strip().strip("`").strip()

    async def _system_aware_merge(
        self,
        candidate_a: ParetoCandidate,
        candidate_b: ParetoCandidate,
    ) -> str:
        strong_a = [k for k, v in candidate_a.metric_scores.items() if v >= candidate_b.metric_scores.get(k, 0) and v > candidate_b.metric_scores.get(k, -1)]
        strong_b = [k for k, v in candidate_b.metric_scores.items() if v >= candidate_a.metric_scores.get(k, 0) and v > candidate_a.metric_scores.get(k, -1)]
        if not strong_a:
            strong_a = list(candidate_a.metric_scores.keys())[:2]
        if not strong_b:
            strong_b = list(candidate_b.metric_scores.keys())[:2]

        prompt = f"""You are a prompt engineer. Two candidate prompts performed differently on metrics.

CANDIDATE A (strong on: {", ".join(strong_a)}):
```
{candidate_a.prompt_text}
```
Scores: {json.dumps(candidate_a.metric_scores)}

CANDIDATE B (strong on: {", ".join(strong_b)}):
```
{candidate_b.prompt_text}
```
Scores: {json.dumps(candidate_b.metric_scores)}

Merge their complementary lessons into a single improved prompt. Return only the merged prompt text, no JSON or extra commentary."""

        raw = await self.improve_llm.generate(
            system_prompt="You are an expert prompt engineer. Return only the merged prompt text.",
            user_prompt=prompt,
        )
        return raw.strip().strip("`").strip()

    def _create_patched_agent(self, new_prompt: str) -> Callable:
        original_prompts = extract_prompts(self.agent_fn, self.agent_file)
        if not original_prompts:
            original_prompts = {"user_prompt_template": new_prompt}
            improved = dict(original_prompts)
        else:
            original_prompts = dict(original_prompts)
            improved = dict(original_prompts)
            if "user_prompt_template" in improved:
                improved["user_prompt_template"] = new_prompt
            else:
                improved["system_prompt"] = new_prompt
        return create_patched_agent(self.agent_fn, original_prompts, improved)

    async def _run_and_evaluate(
        self,
        agent_fn: Callable,
        queries: list[Any],
        metrics: list[MetricFormula],
    ) -> tuple[list[Trace], list[MetricResult], float]:
        traces: list[Trace] = []
        for q in queries:
            t = await run_and_trace(agent_fn, q)
            traces.append(t)
        results = await self.prompt_evaluator.evaluate_batch(traces, metrics)
        composite = self.prompt_evaluator.compute_composite(results)
        return traces, results, composite
