"""
TEI Loop Controller -- 8-Step Flow.

Step 1: Scan agent files, place evaluation checkmarks
Step 2: Baseline evaluation (pre-TEI)
Step 3: Interactive structural fixes (Y/N/Other)
Step 4: Middle evaluation (post-structure-fix)
Step 5: Propose objective metrics (Y/N/Other)
Step 6: Baseline prompt efficiency measurement
Step 7: Iterative prompt optimization (20-50 runs, Pareto front)
Step 8: Final checkmark report (baseline -> middle -> final)
"""

from __future__ import annotations

import asyncio
import datetime
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .models import (
    Dimension,
    DimensionScore,
    EvalResult,
    TEIConfig,
    TEIFullResult,
    Trace,
    Checkpoint,
    CheckpointResult,
    StructuralFix,
    MetricFormula,
    MetricResult,
    ParetoCandidate,
    OptimizationResult,
    RunMode,
    LLMConfig,
)
from .checkpoint_scanner import scan_agent
from .evaluator import TEIEvaluator
from .structural_fixer import StructuralFixer
from .metric_proposer import MetricProposer, format_composite_formula
from .prompt_evaluator import PromptEvaluator
from .prompt_optimizer import PromptOptimizer
from .prompt_improver import extract_prompts
from .tracer import run_and_trace
from .llm_provider import BaseLLMProvider, build_providers, print_model_recommendation
from .adapters.generic import GenericAdapter

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _status_color(status: str) -> str:
    if status == "pass":
        return GREEN
    if status == "weak":
        return YELLOW
    return RED


def _score_to_status(score: float, threshold: float = 0.7) -> str:
    if score >= threshold + 0.15:
        return "pass"
    if score >= threshold:
        return "weak"
    return "fail"


def _dim_label(dim: Dimension) -> str:
    return dim.value.replace("_", " ").title()


class TEILoop:
    """Main TEI Loop: 8-step flow."""

    def __init__(
        self,
        agent: Callable,
        agent_file: Optional[str] = None,
        config: Optional[TEIConfig] = None,
        eval_llm: str = "auto",
        improve_llm: str = "auto",
        provider: str = "auto",
        num_iterations: int = 30,
        verbose: bool = True,
        interactive: bool = True,
    ):
        self.config = config or TEIConfig()

        if provider != "auto":
            self.config.llm.provider = provider
        if eval_llm != "auto":
            self.config.llm.eval_model = eval_llm
        if improve_llm != "auto":
            self.config.llm.improve_model = improve_llm

        self._agent = GenericAdapter(agent)
        self._agent_file = agent_file
        self._num_iterations = num_iterations
        self._verbose = verbose
        self._interactive = interactive

        self._provider_name: Optional[str] = None
        self._eval_llm: Optional[BaseLLMProvider] = None
        self._improve_llm: Optional[BaseLLMProvider] = None
        self._evaluator: Optional[TEIEvaluator] = None
        self._initialized = False
        self._log: list[str] = []
        self._work_dir: Optional[Path] = None
        self._log_path: Optional[Path] = None

    async def run(
        self,
        query: Any,
        test_queries: Optional[list[Any]] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> TEIFullResult:
        """Execute the full 8-step TEI flow."""
        self._init_providers()
        start = time.time()
        result = TEIFullResult()

        work_dir = Path("TEI-work")
        work_dir.mkdir(exist_ok=True)
        self._work_dir = work_dir
        self._log_path = work_dir / f"tei_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        self._log = []

        self._log_and_print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
        self._log_and_print(f"{BOLD}{CYAN}  TEI Loop -- 8-Step Flow{RESET}")
        self._log_and_print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")

        # -------- Step 1 --------
        self._log_and_print(f"{BOLD}Step 1: Scanning agent files...{RESET}")
        agent_files, checkpoints = self._scan_agent()
        result.agent_files = agent_files
        result.checkpoints = checkpoints

        for fp in agent_files:
            p = Path(fp)
            try:
                line_count = len(
                    p.read_text(encoding="utf-8", errors="replace").splitlines()
                )
                self._log_and_print(f"  Found: {p.name} ({line_count} lines)")
            except OSError:
                self._log_and_print(f"  Found: {p.name}")

        if not agent_files:
            self._log_and_print(
                f"  {YELLOW}No agent files found "
                f"(running in black-box mode){RESET}"
            )

        if checkpoints:
            self._log_and_print(
                f"\n  Identified {BOLD}{len(checkpoints)}{RESET} "
                f"checkpoint locations:"
            )
            for idx, cp in enumerate(checkpoints, 1):
                fname = Path(cp.file_path).name
                self._log_and_print(
                    f"    CP-{idx:<3} {fname}:{cp.line_number:<6} "
                    f"{cp.checkpoint_type} ({cp.dimension.value})"
                )
        else:
            self._log_and_print(f"  {DIM}No checkpoints identified.{RESET}")

        self._log_and_print(
            f"\n  {GREEN}Checkmarks placed. "
            f"Ready for baseline evaluation.{RESET}"
        )

        # -------- Step 2 --------
        self._log_and_print(f"\n{BOLD}Step 2: Baseline evaluation (pre-TEI)...{RESET}")
        self._log_and_print("  Running agent with test query...", end="", flush=True)
        trace = await run_and_trace(
            self._agent.agent_fn, query, context=context,
        )
        self._log_and_print(f" done ({trace.total_duration_ms / 1000:.1f}s)")

        self._log_and_print(
            "  Evaluating (4 dimensions in parallel)...",
            end="", flush=True,
        )
        baseline_eval = await self._evaluator.evaluate(trace)
        result.baseline_eval = baseline_eval
        self._log_and_print(" done\n")

        baseline_cp = self._map_checkpoint_results(
            checkpoints, baseline_eval, "baseline",
        )
        result.checkpoint_journey = [baseline_cp]

        self._print_eval_table(
            baseline_eval, "BASELINE EVALUATION (Pre-TEI)",
        )
        self._print_checkpoint_details(checkpoints, baseline_cp)

        if self._work_dir:
            self._save_log(self._work_dir)

        # -------- Step 3 --------
        self._log_and_print(f"\n{BOLD}Step 3: Structural fix proposals...{RESET}")
        fixer = StructuralFixer(self._improve_llm)
        proposed_fixes = await fixer.propose_fixes(
            checkpoints, baseline_cp, baseline_eval, agent_files,
        )

        applied_count = 0
        if not proposed_fixes:
            self._log_and_print(
                f"  {DIM}No structural fixes needed -- "
                f"all checkpoints look healthy.{RESET}"
            )
        else:
            self._log_and_print(
                f"  Based on evaluation, TEI identified "
                f"{BOLD}{len(proposed_fixes)}{RESET} structural issues:\n"
            )
            for i, fix in enumerate(proposed_fixes, 1):
                cp_match = next(
                    (
                        c for c in checkpoints
                        if c.checkpoint_id == fix.checkpoint_id
                    ),
                    None,
                )
                if cp_match:
                    cp_tag = "CP-" + str(checkpoints.index(cp_match) + 1)
                    dim_tag = cp_match.dimension.value
                else:
                    cp_tag = fix.checkpoint_id[:8]
                    dim_tag = "unknown"

                self._log_and_print(
                    f"  {BOLD}[{i}/{len(proposed_fixes)}] "
                    f"{cp_tag} ({dim_tag}){RESET}"
                )
                self._log_and_print(f"        {RED}ISSUE:{RESET} {fix.issue}")
                self._log_and_print(f"        {GREEN}FIX:{RESET}   {fix.proposed_fix}")
                if fix.expected_impact:
                    self._log_and_print(
                        f"        {DIM}Impact: "
                        f"{fix.expected_impact}{RESET}"
                    )

                choice = self._resolve_choice(
                    "        Apply? [Y/N/Other]: ",
                )

                if choice == "y":
                    fix.approved = True
                    if not fix.code_patch:
                        fix = await fixer.generate_fix_details(fix)
                    if fixer.apply_fix(fix):
                        fix.applied = True
                        applied_count += 1
                        self._log_and_print(f"        {GREEN}Applied.{RESET}")
                    else:
                        self._log_and_print(
                            f"        {YELLOW}Could not apply "
                            f"patch automatically.{RESET}"
                        )
                elif choice == "n":
                    self._log_and_print(f"        {DIM}Skipped.{RESET}")
                else:
                    fix.approved = True
                    fix.user_alternative = choice
                    if fixer.apply_fix(fix):
                        fix.applied = True
                        applied_count += 1
                        self._log_and_print(
                            f"        {GREEN}Applied with "
                            f"your modification.{RESET}"
                        )
                    else:
                        self._log_and_print(
                            f"        {YELLOW}Could not apply "
                            f"modification automatically.{RESET}"
                        )

                proposed_fixes[i - 1] = fix
                self._log_and_print("")

            self._log_and_print(
                f"  Structural fixes complete. "
                f"{BOLD}{applied_count} of "
                f"{len(proposed_fixes)}{RESET} applied."
            )

        result.structural_fixes = proposed_fixes

        # -------- Step 4 --------
        middle_eval: Optional[EvalResult] = None

        if applied_count > 0:
            self._log_and_print(
                f"\n{BOLD}Step 4: Middle evaluation "
                f"(post-structure-fix)...{RESET}"
            )
            self._log_and_print(
                "  Re-running agent after structural fixes...",
                end="", flush=True,
            )
            middle_trace = await run_and_trace(
                self._agent.agent_fn, query, context=context,
            )
            self._log_and_print(
                f" done ({middle_trace.total_duration_ms / 1000:.1f}s)"
            )

            self._log_and_print("  Re-evaluating...", end="", flush=True)
            middle_eval = await self._evaluator.evaluate(middle_trace)
            result.middle_eval = middle_eval
            self._log_and_print(" done\n")

            middle_cp = self._map_checkpoint_results(
                checkpoints, middle_eval, "middle",
            )
            result.checkpoint_journey.append(middle_cp)

            self._print_comparison_table(baseline_eval, middle_eval, None)

            delta = (
                middle_eval.aggregate_score - baseline_eval.aggregate_score
            )
            color = GREEN if delta >= 0 else RED
            self._log_and_print(
                f"\n  Structure fixes improved score by "
                f"{color}{delta:+.2f}{RESET}."
            )
            self._log_and_print("  Proceeding to prompt optimization.")
        else:
            self._log_and_print(f"\n{BOLD}Step 4: Middle evaluation{RESET}")
            self._log_and_print(
                f"  {DIM}No structural fixes applied -- "
                f"skipping middle evaluation.{RESET}"
            )
            self._log_and_print("  Proceeding to prompt optimization.")

        if self._work_dir:
            self._save_log(self._work_dir)

        # -------- Step 5: Propose objective metrics --------
        self._log_and_print(f"\n{BOLD}Step 5: Proposing objective metrics (20 iterations, ~100 candidates)...{RESET}")
        prompts_data = extract_prompts(
            self._agent.agent_fn, self._agent_file,
        )
        prompt_text = (
            prompts_data.get("system_prompt", "")
            or prompts_data.get("user_prompt_template", "")
        )

        agent_source = ""
        if self._agent_file:
            try:
                agent_source = Path(self._agent_file).read_text(
                    encoding="utf-8", errors="replace",
                )
            except OSError:
                pass

        proposer = MetricProposer(self._improve_llm)
        all_draft_metrics: list[MetricFormula] = []

        for metric_iter in range(1, 21):
            self._log_and_print(f"  Generating metrics batch {metric_iter}/20...", end="", flush=True)
            batch = await proposer.propose_metrics(
                checkpoints, agent_source, prompt_text,
            )
            all_draft_metrics.extend(batch)
            self._log_and_print(f" got {len(batch)} metrics (total: {len(all_draft_metrics)})")

        # Deduplicate by name (keep first occurrence)
        seen_names = set()
        unique_metrics = []
        for m in all_draft_metrics:
            normalized = m.name.lower().strip()
            if normalized not in seen_names:
                seen_names.add(normalized)
                unique_metrics.append(m)

        self._log_and_print(f"\n  Generated {len(all_draft_metrics)} draft metrics, {len(unique_metrics)} unique after dedup.")

        # Rank by impact: ask LLM to rank which metrics would be most impactful
        # for measuring the VALUE this agent provides
        ranked_metrics = await self._rank_metrics_by_impact(
            unique_metrics, agent_source, prompt_text,
        )

        # Save all draft metrics + rankings to log
        self._log_and_print(f"\n  DRAFT METRICS RANKING ({len(ranked_metrics)} unique metrics):")
        for rank, m in enumerate(ranked_metrics, 1):
            self._log_and_print(f"    #{rank}: {m.name} - {m.description[:80]}")

        # Take top 5
        top_5 = ranked_metrics[:5]
        self._log_and_print(f"\n  Top 5 metrics selected for your approval:\n")

        confirmed_metrics: list[MetricFormula] = []
        if top_5:
            self._log_and_print(
                f"  TEI analyzed your agent and proposes "
                f"{BOLD}{len(top_5)}{RESET} metrics:\n"
            )
            for i, metric in enumerate(top_5, 1):
                self._log_and_print(f"  {BOLD}[{i}] {metric.name}{RESET}")
                self._log_and_print(f"      {metric.description}")
                self._log_and_print(
                    f"      Formula: {CYAN}{metric.formula}{RESET}"
                )
                self._log_and_print(f"      Method:  {metric.measurement_method}")

                choice = self._resolve_choice(
                    "      Accept? [Y/N/Other]: ",
                )

                if choice == "y":
                    metric.approved = True
                    confirmed_metrics.append(metric)
                    self._log_and_print(f"      {GREEN}Accepted.{RESET}")
                elif choice == "n":
                    self._log_and_print(f"      {DIM}Rejected.{RESET}")
                else:
                    metric.approved = True
                    metric.description = (
                        metric.description + " (user: " + choice + ")"
                    )
                    confirmed_metrics.append(metric)
                    self._log_and_print(
                        f"      {GREEN}Accepted with modification.{RESET}"
                    )
                self._log_and_print("")
        else:
            self._log_and_print(
                f"  {YELLOW}No metrics proposed "
                f"(limited agent source available).{RESET}"
            )

        if confirmed_metrics:
            weights = await proposer.propose_composite_weights(
                confirmed_metrics,
            )
            for m in confirmed_metrics:
                if m.name in weights:
                    m.weight = weights[m.name]

            formula_str = format_composite_formula(confirmed_metrics)
            self._log_and_print(f"  {BOLD}COMPOSITE FORMULA:{RESET}")
            self._log_and_print(f"    {CYAN}{formula_str}{RESET}")

            if self._interactive:
                adj = self._prompt_user(
                    "  Adjust weights? [Y/N]: ",
                )
                if adj not in ("y", "n"):
                    self._log_and_print(
                        f"  {DIM}(Custom weight input not yet "
                        f"supported; keeping proposed weights){RESET}"
                    )
            self._log_and_print("")

        result.metrics = confirmed_metrics

        if self._work_dir:
            self._save_log(self._work_dir)

        # -------- Step 6 --------
        self._log_and_print(
            f"\n{BOLD}Step 6: Baseline prompt efficiency "
            f"measurement...{RESET}"
        )
        sample_queries = (test_queries or [])[:5] or [query]
        prompt_eval = PromptEvaluator(self._eval_llm)
        baseline_prompt_scores: list[MetricResult] = []
        baseline_composite = 0.0

        if confirmed_metrics:
            n_samples = len(sample_queries)
            qs = "queries" if n_samples != 1 else "query"
            self._log_and_print(
                f"  Running {n_samples} sample {qs} to measure "
                f"current prompt performance..."
            )
            sample_traces: list[Trace] = []
            for sq in sample_queries:
                t = await run_and_trace(
                    self._agent.agent_fn, sq, context=context,
                )
                sample_traces.append(t)

            baseline_prompt_scores = await prompt_eval.evaluate_batch(
                sample_traces, confirmed_metrics,
            )
            baseline_composite = prompt_eval.compute_composite(
                baseline_prompt_scores,
            )
            result.baseline_prompt_scores = baseline_prompt_scores

            self._print_metric_table(
                baseline_prompt_scores,
                baseline_composite,
                "BASELINE PROMPT EFFICIENCY",
            )
            self._log_and_print(f"\n  This is your starting point.")
            self._log_and_print(
                "  TEI will now optimize prompts to "
                "maximize these metrics."
            )
        else:
            self._log_and_print(
                f"  {DIM}No confirmed metrics -- "
                f"skipping baseline measurement.{RESET}"
            )

        # -------- Step 7 --------
        self._log_and_print(
            f"\n{BOLD}Step 7: Iterative prompt optimization "
            f"(Pareto front)...{RESET}"
        )
        optimization_result: Optional[OptimizationResult] = None

        if confirmed_metrics and prompt_text:
            opt_queries = test_queries if test_queries else [query] * 3
            self._log_and_print(
                f"  Optimizing prompt | Budget: "
                f"{self._num_iterations} iterations"
            )
            self._log_and_print("  Candidate pool starts with 1\n")

            optimizer = PromptOptimizer(
                improve_llm=self._improve_llm,
                eval_llm=self._eval_llm,
                metrics=confirmed_metrics,
                agent_fn=self._agent.agent_fn,
                agent_file=self._agent_file,
            )

            optimization_result = await optimizer.optimize(
                original_prompt=prompt_text,
                test_queries=opt_queries,
                num_iterations=self._num_iterations,
                verbose=self._verbose,
            )
            result.optimization = optimization_result

            self._log_and_print(
                f"\n  All {self._num_iterations} iterations "
                f"completed.\n"
            )
            self._print_pareto_table(
                optimization_result, confirmed_metrics,
            )

            if optimization_result.best_candidate:
                best = optimization_result.best_candidate
                self._log_and_print(
                    f"\n  {BOLD}SELECTED:{RESET} P{best.iteration} "
                    f"(highest composite: "
                    f"{best.composite_score:.1f}%)"
                )
                self._print_optimization_delta(
                    optimization_result, confirmed_metrics,
                )
        else:
            if not confirmed_metrics:
                reason = "no metrics confirmed"
            else:
                reason = "no prompt text extracted"
            self._log_and_print(
                f"  {YELLOW}Skipping optimization "
                f"({reason}).{RESET}"
            )

        # -------- Step 8 --------
        self._log_and_print(f"\n{BOLD}Step 8: Final checkmark report...{RESET}")
        self._log_and_print(
            "  Running final evaluation with optimized prompt...",
            end="", flush=True,
        )

        if (
            optimization_result
            and optimization_result.best_candidate
        ):
            from .prompt_improver import create_patched_agent

            best_prompt = (
                optimization_result.best_candidate.prompt_text
            )
            original_prompts = dict(prompts_data)
            improved_prompts = dict(prompts_data)
            if "system_prompt" in prompts_data:
                improved_prompts["system_prompt"] = best_prompt
            else:
                improved_prompts["user_prompt_template"] = (
                    best_prompt
                )
            patched_fn = create_patched_agent(
                self._agent.agent_fn,
                original_prompts,
                improved_prompts,
            )
            final_trace = await run_and_trace(
                patched_fn, query, context=context,
            )
        else:
            final_trace = await run_and_trace(
                self._agent.agent_fn, query, context=context,
            )

        final_eval = await self._evaluator.evaluate(final_trace)
        result.final_eval = final_eval
        self._log_and_print(" done\n")

        final_cp = self._map_checkpoint_results(
            checkpoints, final_eval, "final",
        )
        result.checkpoint_journey.append(final_cp)

        self._print_comparison_table(
            baseline_eval, middle_eval, final_eval,
        )

        if checkpoints and len(result.checkpoint_journey) >= 2:
            self._log_and_print(f"\n  {BOLD}Checkpoint journey:{RESET}")
            for idx, cp in enumerate(checkpoints):
                stages: list[str] = []
                for stage_results in result.checkpoint_journey:
                    match = next(
                        (
                            r for r in stage_results
                            if r.checkpoint.checkpoint_id
                            == cp.checkpoint_id
                        ),
                        None,
                    )
                    if match:
                        stages.append(match.status.upper())
                    else:
                        stages.append("--")
                journey_str = " -> ".join(stages)
                self._log_and_print(
                    f"    CP-{idx + 1} "
                    f"{cp.checkpoint_type:<25} {journey_str}"
                )

        if (
            confirmed_metrics
            and optimization_result
            and optimization_result.best_candidate
        ):
            self._print_efficiency_journey(
                baseline_prompt_scores,
                baseline_composite,
                optimization_result,
            )

        total_ms = (time.time() - start) * 1000
        result.total_duration_ms = total_ms

        total_cost = 0.0
        if self._eval_llm:
            total_cost += self._eval_llm.get_cost(
                self._provider_name or "openai",
            )
        if self._improve_llm:
            total_cost += self._improve_llm.get_cost(
                self._provider_name or "openai",
            )
        result.total_cost_usd = total_cost

        self._save_results(result, optimization_result, prompts_data)
        self._log_and_print(
            f"  Duration: {total_ms / 1000:.1f}s | "
            f"Cost: ${total_cost:.4f}\n"
        )

        if self._work_dir:
            log_path = self._save_log(self._work_dir)
            self._log_and_print(f"  Log saved to: {log_path}")

        return result

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _log_and_print(self, text: str, end: str = "\n", flush: bool = False) -> None:
        """Print to terminal and append to log (strips ANSI codes for .md)."""
        print(text, end=end, flush=flush)
        clean = re.sub(r'\033\[[0-9;]*m', '', text)
        if end == "":
            if self._log:
                self._log[-1] = self._log[-1] + clean
            else:
                self._log.append(clean)
        else:
            self._log.append(clean)

    def _save_log(self, work_dir: Path, section: str = "") -> str:
        """Save current log to .md file in TEI-work/. Returns the file path."""
        path = self._log_path if self._log_path else work_dir / f"tei_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        content = "# TEI Run Log\n\n"
        content += f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        content += "```\n"
        content += "\n".join(self._log)
        content += "\n```\n"
        path.write_text(content, encoding="utf-8")
        return str(path)

    def _prompt_user(self, prompt_text: str) -> str:
        """Interactive terminal prompt. Returns 'y', 'n', or user text."""
        try:
            raw = input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            return "n"
        lower = raw.lower()
        if lower in ("y", "yes"):
            return "y"
        if lower in ("n", "no"):
            return "n"
        return raw if raw else "n"

    def _resolve_choice(self, prompt_text: str) -> str:
        """Auto-approve when non-interactive, else prompt user."""
        if not self._interactive:
            return "y"
        return self._prompt_user(prompt_text)

    def _init_providers(self) -> None:
        """Initialize LLM providers."""
        if self._initialized:
            return
        (
            self._provider_name,
            self._eval_llm,
            self._improve_llm,
        ) = build_providers(self.config.llm)
        self._evaluator = TEIEvaluator(self.config, self._eval_llm)
        if self.config.show_cost_estimate:
            print_model_recommendation(self.config.llm)
        self._initialized = True

    def _scan_agent(
        self,
    ) -> tuple[list[str], list[Checkpoint]]:
        if self._agent_file:
            return scan_agent(self._agent_file)
        return [], []

    def _map_checkpoint_results(
        self,
        checkpoints: list[Checkpoint],
        eval_result: EvalResult,
        stage: str,
    ) -> list[CheckpointResult]:
        results: list[CheckpointResult] = []
        for cp in checkpoints:
            ds = eval_result.dimension_scores.get(cp.dimension)
            if ds:
                score = ds.score
                status = _score_to_status(score, ds.threshold)
                reasoning = ds.failure_summary or ds.reasoning
            else:
                score = 0.0
                status = "fail"
                reasoning = "Dimension not evaluated"
            results.append(
                CheckpointResult(
                    checkpoint=cp,
                    score=score,
                    status=status,
                    reasoning=reasoning[:300],
                    stage=stage,
                )
            )
        return results

    def _save_results(
        self,
        result: TEIFullResult,
        opt: Optional[OptimizationResult],
        prompts_data: dict[str, Any],
    ) -> None:
        results_dir = Path("tei-results")
        results_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")

        run_path = results_dir / f"run_{ts}.json"
        try:
            run_path.write_text(
                result.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except Exception:
            run_path = None

        if opt and opt.best_candidate:
            prompt_path = results_dir / "optimized_prompts.json"
            payload = {
                "original_prompt": prompts_data,
                "optimized_prompt": (
                    opt.best_candidate.prompt_text
                ),
                "composite_score": (
                    opt.best_candidate.composite_score
                ),
                "metric_scores": (
                    opt.best_candidate.metric_scores
                ),
                "iteration": opt.best_candidate.iteration,
            }
            try:
                prompt_path.write_text(
                    json.dumps(payload, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

        saved = run_path or results_dir
        self._log_and_print(
            f"\n  {GREEN}TEI complete. "
            f"Results saved to: {saved}{RESET}"
        )

    # ------------------------------------------------------------------ #
    #  Printing                                                            #
    # ------------------------------------------------------------------ #

    def _print_eval_table(
        self, eval_result: EvalResult, label: str,
    ) -> None:
        """Print a formatted evaluation table."""
        sep = "  +" + "-" * 26 + "+" + "-" * 9 + "+" + "-" * 10 + "+"
        self._log_and_print(f"  {BOLD}{label}:{RESET}")
        self._log_and_print(sep)
        self._log_and_print(
            f"  | {'Dimension':<24} "
            f"| {'Score':>7} "
            f"| {'Status':<8} |"
        )
        self._log_and_print(sep)
        for dim in Dimension:
            ds = eval_result.dimension_scores.get(dim)
            if not ds:
                continue
            status = _score_to_status(ds.score, ds.threshold)
            color = _status_color(status)
            self._log_and_print(
                f"  | {_dim_label(dim):<24} "
                f"| {ds.score:>5.2f}   "
                f"| {color}{status.upper():<8}{RESET} |"
            )
        self._log_and_print(sep)
        self._log_and_print(
            f"  | {'AGGREGATE':<24} "
            f"| {eval_result.aggregate_score:>5.2f}   "
            f"| {'':8} |"
        )
        self._log_and_print(sep)

    def _print_checkpoint_details(
        self,
        checkpoints: list[Checkpoint],
        cp_results: list[CheckpointResult],
    ) -> None:
        if not cp_results:
            return
        self._log_and_print(f"\n  {BOLD}Checkpoint details:{RESET}")
        for idx, cr in enumerate(cp_results):
            color = _status_color(cr.status)
            summary = cr.reasoning[:60] if cr.reasoning else ""
            self._log_and_print(
                f"    CP-{idx + 1} "
                f"{cr.checkpoint.checkpoint_type:<25} "
                f"{color}{cr.status.upper():<6}{RESET} "
                f"({summary})"
            )

    def _print_comparison_table(
        self,
        baseline: EvalResult,
        middle: Optional[EvalResult],
        final: Optional[EvalResult],
    ) -> None:
        """Print the 3-column comparison table."""
        has_mid = middle is not None
        has_fin = final is not None

        if has_mid and has_fin:
            sep = (
                "  +" + "-" * 26 + "+" + "-" * 10 + "+"
                + "-" * 10 + "+" + "-" * 10 + "+"
                + "-" * 17 + "+"
            )
            self._log_and_print(f"  {BOLD}FINAL CHECKMARK REPORT:{RESET}")
            self._log_and_print(sep)
            self._log_and_print(
                f"  | {'Dimension':<24} "
                f"| {'Baseline':>8} "
                f"| {'Middle':>8} "
                f"| {'Final':>8} "
                f"| {'Total Delta':>15} |"
            )
            self._log_and_print(sep)
            for dim in Dimension:
                bds = baseline.dimension_scores.get(dim)
                mds = middle.dimension_scores.get(dim)
                fds = final.dimension_scores.get(dim)
                bs = bds.score if bds else 0.0
                ms = mds.score if mds else bs
                fs = fds.score if fds else ms
                d = fs - bs
                c = GREEN if d >= 0 else RED
                self._log_and_print(
                    f"  | {_dim_label(dim):<24} "
                    f"| {bs:>6.2f}   "
                    f"| {ms:>6.2f}   "
                    f"| {fs:>6.2f}   "
                    f"| {c}{d:>+13.2f}{RESET}   |"
                )
            self._log_and_print(sep)
            ba = baseline.aggregate_score
            ma = middle.aggregate_score
            fa = final.aggregate_score
            td = fa - ba
            c = GREEN if td >= 0 else RED
            self._log_and_print(
                f"  | {'AGGREGATE':<24} "
                f"| {ba:>6.2f}   "
                f"| {ma:>6.2f}   "
                f"| {fa:>6.2f}   "
                f"| {c}{td:>+13.2f}{RESET}   |"
            )
            self._log_and_print(sep)

        elif has_mid or has_fin:
            after = middle if has_mid else final
            stage_label = "Middle" if has_mid else "Final"
            title = (
                "MIDDLE EVALUATION"
                if has_mid else "FINAL EVALUATION"
            )
            sep = (
                "  +" + "-" * 26 + "+" + "-" * 10 + "+"
                + "-" * 10 + "+" + "-" * 11 + "+"
            )
            self._log_and_print(f"  {BOLD}{title}:{RESET}")
            self._log_and_print(sep)
            self._log_and_print(
                f"  | {'Dimension':<24} "
                f"| {'Before':>8} "
                f"| {stage_label:>8} "
                f"| {'Delta':>9} |"
            )
            self._log_and_print(sep)
            for dim in Dimension:
                bds = baseline.dimension_scores.get(dim)
                ads = after.dimension_scores.get(dim)
                bs = bds.score if bds else 0.0
                asc = ads.score if ads else bs
                d = asc - bs
                c = GREEN if d >= 0 else RED
                self._log_and_print(
                    f"  | {_dim_label(dim):<24} "
                    f"| {bs:>6.2f}   "
                    f"| {asc:>6.2f}   "
                    f"| {c}{d:>+7.2f}{RESET}   |"
                )
            self._log_and_print(sep)
            ba = baseline.aggregate_score
            aa = after.aggregate_score
            d = aa - ba
            c = GREEN if d >= 0 else RED
            self._log_and_print(
                f"  | {'AGGREGATE':<24} "
                f"| {ba:>6.2f}   "
                f"| {aa:>6.2f}   "
                f"| {c}{d:>+7.2f}{RESET}   |"
            )
            self._log_and_print(sep)

    def _print_metric_table(
        self,
        results: list[MetricResult],
        composite: float,
        label: str,
    ) -> None:
        sep = (
            "  +" + "-" * 24 + "+" + "-" * 10 + "+"
            + "-" * 12 + "+"
        )
        self._log_and_print(f"\n  {BOLD}{label}:{RESET}")
        self._log_and_print(sep)
        self._log_and_print(
            f"  | {'Metric':<22} "
            f"| {'Score':>8} "
            f"| {'Detail':<10} |"
        )
        self._log_and_print(sep)
        for r in results:
            detail = r.detail[:10] if r.detail else ""
            self._log_and_print(
                f"  | {r.metric.name:<22} "
                f"| {r.score:>7.1f}% "
                f"| {detail:<10} |"
            )
        self._log_and_print(sep)
        self._log_and_print(
            f"  | {'COMPOSITE (weighted)':<22} "
            f"| {composite:>7.1f}% "
            f"| {'':10} |"
        )
        self._log_and_print(sep)

    def _print_pareto_table(
        self,
        opt_result: OptimizationResult,
        metrics: list[MetricFormula],
    ) -> None:
        front = opt_result.pareto_front
        if not front:
            return

        metric_names = [m.name for m in metrics]
        abbrevs: list[str] = []
        for m in metrics:
            words = m.name.split()[:3]
            ab = "".join(w[0].upper() for w in words)[:4]
            abbrevs.append(ab)

        hdr_parts = ["ID    "]
        for ab in abbrevs:
            hdr_parts.append(f"{ab:>6}")
        hdr_parts.append(f"{'Composite':>10}")
        hdr = " | ".join(hdr_parts)

        rule_parts = ["------"]
        for _ in abbrevs:
            rule_parts.append("------")
        rule_parts.append("----------")
        rule = "-+-".join(rule_parts)

        self._log_and_print(
            f"  {BOLD}PARETO FRONT "
            f"({len(front)} candidates):{RESET}"
        )
        self._log_and_print(f"  +{rule}+")
        self._log_and_print(f"  | {hdr} |")
        self._log_and_print(f"  +{rule}+")

        best = opt_result.best_candidate
        for cand in front:
            parts = [f"P{cand.iteration:<5}"]
            for mn in metric_names:
                val = cand.metric_scores.get(mn, 0)
                parts.append(f"{val:>5.0f}%")
            parts.append(f"{cand.composite_score:>8.1f}%")
            row = " | ".join(parts)
            tag = ""
            if (
                best
                and cand.candidate_id == best.candidate_id
            ):
                tag = f"  {BOLD}<-- BEST{RESET}"
            self._log_and_print(f"  | {row} |{tag}")

        self._log_and_print(f"  +{rule}+")

    def _print_optimization_delta(
        self,
        opt_result: OptimizationResult,
        metrics: list[MetricFormula],
    ) -> None:
        sep = (
            "  +" + "-" * 24 + "+" + "-" * 10 + "+"
            + "-" * 10 + "+" + "-" * 11 + "+"
        )
        self._log_and_print(f"\n  {BOLD}OPTIMIZATION RESULT:{RESET}")
        self._log_and_print(sep)
        self._log_and_print(
            f"  | {'Metric':<22} "
            f"| {'Before':>8} "
            f"| {'After':>8} "
            f"| {'Delta':>9} |"
        )
        self._log_and_print(sep)

        w_before = 0.0
        w_after = 0.0
        for m in metrics:
            before = opt_result.baseline_scores.get(m.name, 0)
            after = opt_result.final_scores.get(m.name, 0)
            d = after - before
            c = GREEN if d >= 0 else RED
            w_before += before * m.weight
            w_after += after * m.weight
            self._log_and_print(
                f"  | {m.name:<22} "
                f"| {before:>7.0f}% "
                f"| {after:>7.0f}% "
                f"| {c}{d:>+8.0f}%{RESET} |"
            )

        self._log_and_print(sep)
        tw = sum(m.weight for m in metrics) or 1.0
        cb = w_before / tw
        ca = w_after / tw
        cd = ca - cb
        c = GREEN if cd >= 0 else RED
        self._log_and_print(
            f"  | {'COMPOSITE':<22} "
            f"| {cb:>7.1f}% "
            f"| {ca:>7.1f}% "
            f"| {c}{cd:>+8.1f}%{RESET} |"
        )
        self._log_and_print(sep)

    def _print_efficiency_journey(
        self,
        baseline_scores: list[MetricResult],
        baseline_composite: float,
        opt: OptimizationResult,
    ) -> None:
        self._log_and_print(f"\n  {BOLD}Prompt efficiency journey:{RESET}")
        sep = (
            "  +" + "-" * 24 + "+" + "-" * 10 + "+"
            + "-" * 10 + "+" + "-" * 11 + "+"
        )
        self._log_and_print(sep)
        self._log_and_print(
            f"  | {'Metric':<22} "
            f"| {'Before':>8} "
            f"| {'After':>8} "
            f"| {'Delta':>9} |"
        )
        self._log_and_print(sep)
        for bp in baseline_scores:
            name = bp.metric.name
            before = bp.score
            after = opt.final_scores.get(name, before)
            d = after - before
            c = GREEN if d >= 0 else RED
            self._log_and_print(
                f"  | {name:<22} "
                f"| {before:>7.1f}% "
                f"| {after:>7.1f}% "
                f"| {c}{d:>+8.1f}%{RESET} |"
            )
        self._log_and_print(sep)
        bc = baseline_composite
        ac = opt.best_candidate.composite_score
        dc = ac - bc
        c = GREEN if dc >= 0 else RED
        self._log_and_print(
            f"  | {'COMPOSITE':<22} "
            f"| {bc:>7.1f}% "
            f"| {ac:>7.1f}% "
            f"| {c}{dc:>+8.1f}%{RESET} |"
        )
        self._log_and_print(sep)

    # ------------------------------------------------------------------ #
    #  Legacy methods (kept for backward compatibility)                     #
    # ------------------------------------------------------------------ #

    async def evaluate_only(
        self,
        query: Any,
        context: Optional[dict[str, Any]] = None,
    ) -> EvalResult:
        """[Legacy] Run agent once and evaluate without improvement."""
        self._init_providers()
        trace = await run_and_trace(
            self._agent.agent_fn, query, context=context,
        )
        return await self._evaluator.evaluate(trace)

    async def compare(
        self,
        query: Any,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, EvalResult]:
        """[Legacy] Run baseline + full TEI loop, return both."""
        baseline = await self.evaluate_only(query, context)
        full = await self.run(query, context=context)
        return {
            "baseline": baseline,
            "final": full.final_eval,
        }
