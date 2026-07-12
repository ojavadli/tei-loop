from .models import MetricFormula, MetricResult, ParetoCandidate, OptimizationResult, Trace
from .llm_provider import BaseLLMProvider
from .pareto import update_pareto_front, sample_from_front, sample_pair_from_front, compute_composite, select_best
from .prompt_evaluator import PromptEvaluator
from .tracer import run_and_trace
from .prompt_improver import extract_prompts, create_patched_agent
import random
import json
import time
from typing import Any, Callable, Optional

VALID_OPTIMIZER_MODES = ("pareto", "cubic", "hybrid")


class PromptOptimizer:
    def __init__(
        self,
        improve_llm: BaseLLMProvider,
        eval_llm: BaseLLMProvider,
        metrics: list[MetricFormula],
        agent_fn: Callable,
        agent_file: Optional[str] = None,
        optimizer_mode: str = "pareto",
        cubic_config: Optional[Any] = None,   # tei_loop.cubic.CubicConfig
    ):
        if optimizer_mode not in VALID_OPTIMIZER_MODES:
            raise ValueError(
                f"optimizer_mode must be one of {VALID_OPTIMIZER_MODES}, "
                f"got {optimizer_mode!r}")
        self.improve_llm = improve_llm
        self.eval_llm = eval_llm
        self.metrics = metrics
        self.agent_fn = agent_fn
        self.agent_file = agent_file
        self.prompt_evaluator = PromptEvaluator(eval_llm)
        self.rng = random.Random(42)
        self.optimizer_mode = optimizer_mode
        self.cubic_config = cubic_config

    async def optimize(
        self,
        original_prompt: str,
        test_queries: list[Any],
        num_iterations: int = 30,
        verbose: bool = True,
    ) -> OptimizationResult:
        if self.optimizer_mode in ("cubic", "hybrid"):
            return await self._optimize_cubic(
                original_prompt, test_queries, num_iterations, verbose)
        return await self._optimize_pareto(
            original_prompt, test_queries, num_iterations, verbose)

    async def _optimize_pareto(
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
        saved_baseline_scores = dict(p0.metric_scores)

        for i in range(1, num_iterations + 1):
            use_merge = len(front) >= 2 and self.rng.random() < 0.30
            if use_merge:
                ca, cb = sample_pair_from_front(front, self.rng)
                new_prompt = await self._system_aware_merge(ca, cb)
                strategy_name = "merge"
                parent_a, parent_b = ca, cb
            else:
                parent = sample_from_front(front, self.rng)
                trace = self._pick_trace_for_candidate(parent, traces)
                parent_results = [MetricResult(metric=m, score=parent.metric_scores.get(m.name, 0), detail="") for m in self.metrics]
                new_prompt = await self._reflective_mutation(parent, trace, parent_results)
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
                abbrevs = self._metric_abbrevs(metric_scores_raw, front[0].metric_scores if front else {})
                delta_str = ", ".join(abbrevs) if abbrevs else ""
                add_str = f" new Pareto candidate ({delta_str})" if added else ""
                ref_preview = ""
                if strategy_name == "mutation":
                    ref_preview = new_prompt[:80].replace("\n", " ") + "..." if len(new_prompt) > 80 else new_prompt[:80]
                else:
                    ref_preview = f"merged from {parent_a.iteration} + {parent_b.iteration}"
                print(f"  Iter {i:2}/{num_iterations} | Comp: {comp_new:.1f}% | Pool: {len(front)}{add_str}")
                print(f"    {strategy_name.capitalize()} from P{parent_a.iteration}. {ref_preview}")

        best = select_best(front)
        return OptimizationResult(
            total_iterations=num_iterations,
            pareto_front=front,
            best_candidate=best,
            metric_history=metric_history,
            baseline_scores=saved_baseline_scores,
            final_scores=best.metric_scores,
            optimizer_mode="pareto",
            final_selection_source="pareto_composite",
        )

    # ------------------------------------------------------------------ #
    #  D-ARC (Discrete Adaptive Cubic Regularization) modes               #
    # ------------------------------------------------------------------ #

    async def _propose_cubic_candidates(
        self,
        incumbent_prompt: str,
        sigma: float,
        radius: float,
        recent_history: list[dict],
        weakest: list[str],
        n_proposals: int,
        iteration: int,
    ) -> list:
        """One optimizer API call returning up to n_proposals distinct prompt
        proposals spanning micro/small/medium/large edits (structured JSON).
        The per-proposal `rationale` is diagnostic metadata only — it is never
        injected into the task agent's prompt."""
        from .cubic import PromptProposal

        hist_lines = []
        for h in recent_history[-6:]:
            hist_lines.append(
                f"- iter {h.get('iteration')}: {h.get('outcome')} "
                f"(scale={h.get('requested_scale')}, "
                f"dU={-(h.get('actual_reduction') or 0.0):+.4f})")
        metric_lines = [f"- {m.name}: {m.description} (formula: {m.formula})"
                        for m in self.metrics]

        user = f"""You are improving an agent's prompt inside a trust-region-style loop.

CURRENT PROMPT:
```
{incumbent_prompt}
```

METRICS BEING MAXIMIZED:
{chr(10).join(metric_lines)}

WEAKEST METRICS RIGHT NOW: {", ".join(weakest) if weakest else "(unknown)"}

RECENT STEP HISTORY (accepted/rejected edits):
{chr(10).join(hist_lines) if hist_lines else "(none yet)"}

CURRENT REGULARIZATION sigma = {sigma:.4g}. TARGET EDIT RADIUS = {radius:.2f}
(0 = change almost nothing, 1 = free rewrite). Respect the radius: with a
small radius propose conservative edits; with a large radius bolder ones.

Return STRICT JSON:
{{"proposals": [
  {{"prompt": "<full improved prompt text>",
    "requested_scale": "micro|small|medium|large",
    "rationale": "<one concise sentence: what was changed and why>"}},
  ... up to {n_proposals} DISTINCT proposals, approximately spanning the
  micro, small, medium and large edit scales ...
]}}
(variation id: cubic-{iteration})"""

        raw = await self.improve_llm.generate_json(
            system_prompt=("You are an expert prompt engineer. Return only valid "
                           "JSON with a 'proposals' array. Each proposal must be a "
                           "complete standalone prompt, not a diff."),
            user_prompt=user,
        )
        proposals = []
        for item in (raw.get("proposals") or [])[: n_proposals]:
            if not isinstance(item, dict):
                continue
            proposals.append(PromptProposal(
                prompt=str(item.get("prompt", "")),
                requested_scale=str(item.get("requested_scale", "medium")).lower(),
                rationale=str(item.get("rationale", "")),
            ))
        return proposals

    async def _optimize_cubic(
        self,
        original_prompt: str,
        test_queries: list[Any],
        num_iterations: int = 30,
        verbose: bool = True,
    ) -> OptimizationResult:
        """cubic:  D-ARC incumbent selection on the scalar loss F = 1 - U.
        hybrid: D-ARC controls acceptance/sigma; a Pareto archive keeps every
        evaluated candidate's raw metric vector and supplies final options.

        Every candidate is evaluated on the SAME fixed search set
        (test_queries, in full) — surrogate observations never mix
        minibatches, models, splits, or scoring configurations.
        """
        from .cubic import (CubicConfig, DiscreteARCController, scalar_utility)

        mode = self.optimizer_mode
        cfg = self.cubic_config or CubicConfig()
        weights = self._metric_weights()
        context_id = (
            f"tei-loop|{mode}|improve={getattr(self.improve_llm, 'model', '?')}"
            f"|eval={getattr(self.eval_llm, 'model', '?')}"
            f"|metrics={','.join(sorted(m.name for m in self.metrics))}"
            f"|split=fixed[{len(test_queries)}]"
        )
        controller = DiscreteARCController(cfg, context_id=context_id)

        async def _eval_full(prompt: str):
            patched = self._create_patched_agent(prompt)
            traces, results, _ = await self._run_and_evaluate(
                patched, test_queries, self.metrics)
            raw = {r.metric.name: r.score for r in results}
            u = scalar_utility(raw, weights)
            return raw, u, results

        raw0, u0, _ = await _eval_full(original_prompt)
        f0 = 1.0 - u0
        controller.register_baseline(original_prompt, F=f0, U=u0)

        p0 = ParetoCandidate(
            iteration=0, prompt_text=original_prompt, metric_scores=raw0,
            composite_score=compute_composite(
                {k: v / 100.0 for k, v in raw0.items()}, weights),
            strategy="baseline", reflection="",
        )
        front = [p0]
        metric_history = [dict(raw0)]
        candidates_by_sha = {controller.incumbent["sha"]: p0}

        if verbose:
            print(f"  [D-ARC {mode}] baseline U={u0:.4f} F={f0:.4f} "
                  f"sigma={controller.sigma:.3g} (fixed search set: "
                  f"{len(test_queries)} queries)")

        i = 0
        while i < num_iterations and not controller.should_stop:
            i += 1
            inc_cand = candidates_by_sha.get(controller.incumbent["sha"], p0)
            weakest = [
                m.name for m in sorted(
                    self.metrics,
                    key=lambda m: inc_cand.metric_scores.get(m.name, 0.0))
            ][:2]
            selection = None
            for attempt in (1, 2):    # retry the proposal call once if empty
                proposals = await self._propose_cubic_candidates(
                    controller.incumbent["prompt"], controller.sigma,
                    controller.target_radius(),
                    controller.history_dicts(), weakest,
                    cfg.proposals_per_iteration, iteration=i * 10 + attempt,
                )
                selection = controller.select_proposal(proposals)
                if selection["prompt"] is not None:
                    break
            if selection is None or selection["prompt"] is None:
                rec = controller.observe_failure(
                    "no valid proposal after retry")
                if verbose:
                    print(f"  Iter {i:2}/{num_iterations} [D-ARC {mode}] "
                          f"no valid proposal (sigma->{controller.sigma:.3g})")
                metric_history.append({})
                continue

            try:
                raw, u, _res = await _eval_full(selection["prompt"])
            except Exception as e:  # evaluation failure: sigma up, state intact
                rec = controller.observe_failure(f"evaluation failed: {e}")
                if verbose:
                    print(f"  Iter {i:2}/{num_iterations} [D-ARC {mode}] "
                          f"evaluation failed ({e}); sigma->{controller.sigma:.3g}")
                metric_history.append({})
                continue

            rec = controller.observe(selection["prompt"], F=1.0 - u, U=u)

            cand = ParetoCandidate(
                iteration=i, prompt_text=selection["prompt"], metric_scores=raw,
                composite_score=compute_composite(
                    {k: v / 100.0 for k, v in raw.items()}, weights),
                strategy=f"darc_{rec.phase}", reflection="",
            )
            candidates_by_sha[rec.prompt_sha] = cand
            # every evaluated proposal — accepted or rejected — is offered to
            # the Pareto archive with its raw metric vector
            front = update_pareto_front(front, cand)
            metric_history.append(dict(raw))

            if verbose:
                word = ("step accepted" if rec.accepted else
                        "step rejected" if rec.phase == "cubic" else rec.phase)
                print(f"  Iter {i:2}/{num_iterations} [D-ARC {mode}] {word} "
                      f"({rec.outcome}) U={u:.4f} rho="
                      f"{'n/a' if rec.rho is None else f'{rec.rho:.2f}'} "
                      f"sigma={controller.sigma:.3g} archive={len(front)}")

        # ---- final selection -------------------------------------------------
        if mode == "cubic":
            winner = controller.best_incumbent()
            best = candidates_by_sha.get(winner["sha"], p0)
            selection_source = "cubic_best_accepted_incumbent"
        else:  # hybrid: rank non-dominated candidates by scalar utility
            def _u_of(c: ParetoCandidate) -> float:
                return scalar_utility(c.metric_scores, weights)
            best = max(front, key=_u_of)
            selection_source = "hybrid_pareto_scalar_utility_rank"

        u_best = scalar_utility(best.metric_scores, weights)
        return OptimizationResult(
            total_iterations=i,
            pareto_front=front,
            best_candidate=best,
            metric_history=metric_history,
            baseline_scores=dict(raw0),
            final_scores=best.metric_scores,
            optimizer_mode=mode,
            cubic_config=cfg.to_dict(),
            cubic_history=controller.history_dicts(),
            incumbent_history=list(controller.incumbent_history),
            surrogate_diagnostics=[
                r.surrogate for r in controller.records if r.surrogate],
            final_selection_source=selection_source,
            scalar_utility_baseline=u0,
            scalar_utility_final=u_best,
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

    def _pick_trace_for_candidate(self, candidate: ParetoCandidate, traces: list[Trace]) -> Trace:
        if not traces:
            return Trace(agent_input=None, agent_output=None)
        return traces[self.rng.randint(0, len(traces) - 1)]

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
            original_prompts = {}
        original_copy = dict(original_prompts)
        improved = dict(original_prompts)
        if "system_prompt" in improved:
            improved["system_prompt"] = new_prompt
        elif "user_prompt_template" in improved:
            improved["user_prompt_template"] = new_prompt
        else:
            improved["system_prompt"] = new_prompt
        return create_patched_agent(self.agent_fn, original_copy, improved, agent_file=self.agent_file)

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
