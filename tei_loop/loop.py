"""
TEI Loop Controller -- 8-Step Flow.

Step 1: Scan agent files, place evaluation checkmarks
Step 2: Baseline evaluation (pre-TEI)
Step 3: Iterative structural fixes (20-iteration batches, Y/N/Other)
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
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Optional
from pathlib import Path

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


def _checkpoint_status(score: float) -> str:
    """Fixed-threshold status for checkpoint mapping: >=0.90 pass, >=0.70 weak, else fail."""
    if score >= 0.90:
        return "pass"
    if score >= 0.70:
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
        self._original_agent_file = agent_file
        self._num_iterations = num_iterations
        self._verbose = verbose
        self._interactive = interactive

        self._provider_name: Optional[str] = None
        self._eval_llm: Optional[BaseLLMProvider] = None
        self._improve_llm: Optional[BaseLLMProvider] = None
        self._evaluator: Optional[TEIEvaluator] = None
        self._work_dir: Optional[Path] = None
        self._initialized = False
        self._log: list[str] = []

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

        print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
        print(f"{BOLD}{CYAN}  TEI Loop -- 8-Step Flow{RESET}")
        print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")

        # -------- Copy agent to TEI-work/ before starting --------
        if self._agent_file:
            agent_path = Path(self._agent_file).resolve()
            work_dir = agent_path.parent / "TEI-work"
            work_dir.mkdir(parents=True, exist_ok=True)
            ts_label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"  Copying agent files to TEI-work/ ({ts_label})...")

            pre_scan_files, _ = scan_agent(self._agent_file)
            if not pre_scan_files:
                pre_scan_files = [str(agent_path)]

            for fp in pre_scan_files:
                src = Path(fp).resolve()
                dst = work_dir / src.name
                shutil.copy2(str(src), str(dst))
                print(f"    Copied: {src.name}")

            self._original_agent_file = self._agent_file
            self._agent_file = str(work_dir / agent_path.name)
            self._work_dir = work_dir
            print(f"  Working copy: {self._agent_file}")
            print(f"  {GREEN}Originals are safe. TEI operates on the copy.{RESET}\n")

        # -------- Step 1: Scan agent files --------
        print(f"{BOLD}Step 1: Scanning agent files...{RESET}")
        agent_files, checkpoints = self._scan_agent()
        result.agent_files = agent_files
        result.checkpoints = checkpoints

        for fp in agent_files:
            p = Path(fp)
            try:
                line_count = len(
                    p.read_text(encoding="utf-8", errors="replace").splitlines()
                )
                print(f"  Found: {p.name} ({line_count} lines)")
            except OSError:
                print(f"  Found: {p.name}")

        if not agent_files:
            print(f"  {YELLOW}No agent files found (running in black-box mode){RESET}")

        if checkpoints:
            print(f"\n  Identified {BOLD}{len(checkpoints)}{RESET} checkpoint locations:")
            for idx, cp in enumerate(checkpoints, 1):
                fname = Path(cp.file_path).name
                print(
                    f"    CP-{idx:<3} {fname}:{cp.line_number:<6} "
                    f"{cp.checkpoint_type} ({cp.dimension.value})"
                )
        else:
            print(f"  {DIM}No checkpoints identified.{RESET}")

        print(f"\n  {GREEN}Checkmarks placed. Ready for baseline evaluation.{RESET}")

        # -------- Step 2: Baseline evaluation --------
        print(f"\n{BOLD}Step 2: Baseline evaluation (pre-TEI)...{RESET}")
        print(f"  Running agent with test query...", end="", flush=True)
        trace = await run_and_trace(self._agent.agent_fn, query, context=context)
        print(f" done ({trace.total_duration_ms / 1000:.1f}s)")

        print(f"  Evaluating (4 dimensions in parallel)...", end="", flush=True)
        baseline_eval = await self._evaluator.evaluate(trace)
        result.baseline_eval = baseline_eval
        print(f" done\n")

        baseline_cp = self._map_checkpoint_results(checkpoints, baseline_eval, "baseline")
        result.checkpoint_journey = [baseline_cp]

        self._print_eval_table(baseline_eval, "BASELINE EVALUATION (Pre-TEI)")
        self._print_checkpoint_details(checkpoints, baseline_cp)

        # -------- Step 3 + 4: Iterative structural fixes (20-iteration batches) --------
        print(f"\n{BOLD}Step 3: Iterative structural fixes (20-iteration batches)...{RESET}")
        fixer = StructuralFixer(self._improve_llm)
        total_applied = 0
        all_fixes: list[StructuralFix] = []
        current_eval = baseline_eval
        current_cp = baseline_cp
        middle_eval: Optional[EvalResult] = None
        batch_number = 0
        best_eval = baseline_eval
        best_fix_description = ""

        print(
            f"  Baseline aggregate: {current_eval.aggregate_score:.3f}\n"
        )

        file_snapshots: dict[str, str] = {}
        for fp in agent_files:
            try:
                file_snapshots[fp] = Path(fp).read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
        best_snapshots = dict(file_snapshots)

        while True:
            batch_number += 1
            batch_applied = 0
            batch_log: list[tuple[int, float, float, str]] = []

            print(
                f"  {BOLD}--- Batch {batch_number} "
                f"(iterations {(batch_number - 1) * 20 + 1}"
                f"-{batch_number * 20}) ---{RESET}"
            )

            for i in range(1, 21):
                iteration_num = (batch_number - 1) * 20 + i

                dim_summary = "  ".join(
                    f"{_dim_label(d)}={ds.score:.2f}"
                    for d, ds in sorted(
                        current_eval.dimension_scores.items(),
                        key=lambda kv: kv[1].score,
                    )
                )

                print(
                    f"    [{iteration_num:>3}] All dims ({dim_summary})...",
                    end="", flush=True,
                )

                proposed = await fixer.propose_holistic_fix(
                    checkpoints, current_cp, current_eval, agent_files,
                )

                if not proposed:
                    print(f" no fix generated")
                    batch_log.append((
                        iteration_num, current_eval.aggregate_score,
                        current_eval.aggregate_score - baseline_eval.aggregate_score,
                        "no_fix",
                    ))
                    continue

                fix = proposed[0]
                fix.approved = True
                if not fix.code_patch:
                    fix = await fixer.generate_fix_details(fix)

                pre_patch_snapshots: dict[str, str] = {}
                for fp in agent_files:
                    try:
                        pre_patch_snapshots[fp] = Path(fp).read_text(
                            encoding="utf-8", errors="replace"
                        )
                    except OSError:
                        pass

                applied_ok = False
                if fix.code_patch and fixer.apply_fix(fix):
                    fix.applied = True
                    applied_ok = True
                    batch_applied += 1
                    total_applied += 1

                all_fixes.append(fix)

                reverted_this = False
                if applied_ok:
                    iter_trace = await run_and_trace(
                        self._agent.agent_fn, query, context=context,
                    )
                    current_eval = await self._evaluator.evaluate(iter_trace)
                    current_cp = self._map_checkpoint_results(
                        checkpoints, current_eval, "middle",
                    )

                    if current_eval.aggregate_score > best_eval.aggregate_score:
                        best_eval = current_eval
                        best_fix_description = fix.proposed_fix
                        for fp in agent_files:
                            try:
                                best_snapshots[fp] = Path(fp).read_text(
                                    encoding="utf-8", errors="replace"
                                )
                            except OSError:
                                pass
                    else:
                        reverted_this = True
                        for fp, content in pre_patch_snapshots.items():
                            try:
                                Path(fp).write_text(content, encoding="utf-8")
                            except OSError:
                                pass
                        current_eval = best_eval
                        current_cp = self._map_checkpoint_results(
                            checkpoints, current_eval, "middle",
                        )

                delta = current_eval.aggregate_score - baseline_eval.aggregate_score
                color = GREEN if delta > 0 else YELLOW if delta == 0 else RED
                if reverted_this:
                    tag = "reverted"
                elif applied_ok:
                    tag = "applied"
                else:
                    tag = "no_patch"
                print(
                    f" {tag} | {current_eval.aggregate_score:.3f} "
                    f"({color}{delta:+.3f}{RESET})"
                )

                batch_log.append((
                    iteration_num,
                    current_eval.aggregate_score,
                    delta,
                    tag,
                ))

            if batch_log:
                self._print_batch_summary(batch_number, batch_log, batch_applied)

            for fp, content in best_snapshots.items():
                try:
                    Path(fp).write_text(content, encoding="utf-8")
                except OSError:
                    pass
            current_eval = best_eval
            middle_eval = current_eval

            batch_delta = middle_eval.aggregate_score - baseline_eval.aggregate_score
            best_delta = best_eval.aggregate_score - baseline_eval.aggregate_score
            color = GREEN if batch_delta > 0 else YELLOW if batch_delta == 0 else RED

            print(
                f"\n  {BOLD}Step 4: Middle evaluation "
                f"(after batch {batch_number}){RESET}"
            )
            self._print_comparison_table(baseline_eval, middle_eval, None)

            if best_delta > 0 and best_fix_description:
                print(
                    f"\n  {GREEN}Best improvement: {best_delta:+.3f} "
                    f"(score {best_eval.aggregate_score:.3f}){RESET}"
                )
                print(
                    f"  {DIM}Best fix: {best_fix_description[:120]}{RESET}"
                )

            print(f"\n  Total delta from baseline: {color}{batch_delta:+.3f}{RESET}")

            if not self._interactive:
                print(f"  Non-interactive mode: continuing to Step 5...")
                break
            user_choice = self._prompt_user(
                f'\n  Run 20 more iterations? (Say "N" to continue to Step 5): '
            )
            if user_choice == "n":
                print(f"  Continuing to Step 5...")
                break
            else:
                print(f"  Running 20 more iterations...\n")
                continue

        result.structural_fixes = all_fixes

        if middle_eval is not None:
            result.middle_eval = middle_eval
            middle_cp_final = self._map_checkpoint_results(
                checkpoints, middle_eval, "middle",
            )
            result.checkpoint_journey.append(middle_cp_final)
        else:
            print(f"\n{BOLD}Step 4: Middle evaluation{RESET}")
            print(f"  {DIM}No structural fixes applied -- skipping.{RESET}")

        # -------- Step 5: Propose objective metrics --------
        print(f"\n{BOLD}Step 5: Proposing objective metrics...{RESET}")
        prompts_data = extract_prompts(self._agent.agent_fn, self._agent_file)
        prompt_text = prompts_data.get("system_prompt", "") or prompts_data.get(
            "user_prompt_template", "",
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
        proposed_metrics = await proposer.propose_metrics(
            checkpoints, agent_source, prompt_text,
        )

        confirmed_metrics: list[MetricFormula] = []
        if proposed_metrics:
            print(
                f"  TEI analyzed your agent and proposes "
                f"{BOLD}{len(proposed_metrics)}{RESET} metrics:\n"
            )
            for i, metric in enumerate(proposed_metrics, 1):
                print(f"  {BOLD}[{i}] {metric.name}{RESET}")
                print(f"      {metric.description}")
                print(f"      Formula: {CYAN}{metric.formula}{RESET}")
                print(f"      Method:  code_based")

                choice = self._resolve_choice(f"      Accept? [Y/N/Other]: ")

                if choice == "y":
                    metric.approved = True
                    confirmed_metrics.append(metric)
                    print(f"      {GREEN}Accepted.{RESET}")
                elif choice == "n":
                    print(f"      {DIM}Rejected.{RESET}")
                else:
                    metric.approved = True
                    metric.description = metric.description + " (user: " + choice + ")"
                    confirmed_metrics.append(metric)
                    print(f"      {GREEN}Accepted with modification.{RESET}")
                print()
        else:
            print(f"  {YELLOW}No metrics proposed (limited agent source available).{RESET}")

        if confirmed_metrics:
            weights = await proposer.propose_composite_weights(confirmed_metrics)
            for m in confirmed_metrics:
                if m.name in weights:
                    m.weight = weights[m.name]

            formula_str = format_composite_formula(confirmed_metrics)
            print(f"  {BOLD}COMPOSITE FORMULA:{RESET}")
            print(f"    {CYAN}{formula_str}{RESET}")

            if self._interactive:
                adj = self._prompt_user("  Adjust weights? [Y/N]: ")
                if adj not in ("y", "n"):
                    print(
                        f"  {DIM}(Custom weight input not yet supported; "
                        f"keeping proposed weights){RESET}"
                    )
            print()

        result.metrics = confirmed_metrics

        # -------- Step 6: Baseline prompt efficiency --------
        print(f"\n{BOLD}Step 6: Baseline prompt efficiency measurement...{RESET}")
        sample_queries = (test_queries or [])[:5] or [query]
        prompt_eval = PromptEvaluator(self._eval_llm)
        baseline_prompt_scores: list[MetricResult] = []
        baseline_composite = 0.0

        if confirmed_metrics:
            n_samples = len(sample_queries)
            qs = "queries" if n_samples != 1 else "query"
            print(
                f"  Running {n_samples} sample {qs} to measure "
                f"current prompt performance..."
            )
            sample_traces: list[Trace] = []
            for sq in sample_queries:
                t = await run_and_trace(self._agent.agent_fn, sq, context=context)
                sample_traces.append(t)

            baseline_prompt_scores = await prompt_eval.evaluate_batch(
                sample_traces, confirmed_metrics,
            )
            baseline_composite = prompt_eval.compute_composite(baseline_prompt_scores)
            result.baseline_prompt_scores = baseline_prompt_scores

            self._print_metric_table(
                baseline_prompt_scores, baseline_composite,
                "BASELINE PROMPT EFFICIENCY",
            )
            print(f"\n  This is your starting point.")
            print(f"  TEI will now optimize prompts to maximize these metrics.")
        else:
            print(f"  {DIM}No confirmed metrics -- skipping baseline measurement.{RESET}")

        # -------- Step 7: Iterative prompt optimization (Pareto front) --------
        print(f"\n{BOLD}Step 7: Iterative prompt optimization (Pareto front)...{RESET}")
        optimization_result: Optional[OptimizationResult] = None

        if confirmed_metrics and prompt_text:
            opt_queries = test_queries if test_queries else [query] * 3
            print(f"  Optimizing prompt | Budget: {self._num_iterations} iterations")
            print(f"  Candidate pool starts with 1\n")

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

            print(f"\n  All {self._num_iterations} iterations completed.\n")
            self._print_pareto_table(optimization_result, confirmed_metrics)

            if optimization_result.best_candidate:
                best = optimization_result.best_candidate
                print(
                    f"\n  {BOLD}SELECTED:{RESET} P{best.iteration} "
                    f"(highest composite: {best.composite_score:.1f}%)"
                )
                self._print_optimization_delta(optimization_result, confirmed_metrics)
        else:
            reason = (
                "no metrics confirmed" if not confirmed_metrics
                else "no prompt text extracted"
            )
            print(f"  {YELLOW}Skipping optimization ({reason}).{RESET}")

        # -------- Step 8: Final checkmark report --------
        print(f"\n{BOLD}Step 8: Final checkmark report...{RESET}")

        use_optimized = False
        if (
            optimization_result
            and optimization_result.best_candidate
            and optimization_result.best_candidate.composite_score > baseline_composite
        ):
            use_optimized = True

        if use_optimized:
            print(f"  Running final evaluation with optimized prompt...", end="", flush=True)
            from .prompt_improver import create_patched_agent

            best_prompt = optimization_result.best_candidate.prompt_text
            original_prompts = dict(prompts_data)
            improved_prompts = dict(prompts_data)
            if "system_prompt" in prompts_data:
                improved_prompts["system_prompt"] = best_prompt
            else:
                improved_prompts["user_prompt_template"] = best_prompt
            patched_fn = create_patched_agent(
                self._agent.agent_fn, original_prompts, improved_prompts,
                agent_file=self._agent_file,
            )
            final_trace = await run_and_trace(patched_fn, query, context=context)

            if self._work_dir:
                opt_path = self._work_dir / "optimized_prompt.txt"
                opt_path.write_text(best_prompt, encoding="utf-8")
                print(f"  Optimized prompt saved to: TEI-work/optimized_prompt.txt")
                print(f"  {DIM}Original agent files are untouched.{RESET}")
        else:
            if optimization_result and optimization_result.best_candidate:
                print(
                    f"  {YELLOW}Optimized prompt did not improve composite score "
                    f"({optimization_result.best_candidate.composite_score:.1f}% vs "
                    f"{baseline_composite:.1f}% baseline). Keeping original.{RESET}"
                )
            print(f"  Running final evaluation with current agent...", end="", flush=True)
            final_trace = await run_and_trace(
                self._agent.agent_fn, query, context=context,
            )

        final_eval = await self._evaluator.evaluate(final_trace)

        reference_eval = middle_eval if middle_eval else baseline_eval
        if use_optimized and final_eval.aggregate_score < reference_eval.aggregate_score:
            print(f" done")
            print(
                f"  {YELLOW}Optimized prompt regressed 4-dimension score "
                f"({final_eval.aggregate_score:.3f} vs {reference_eval.aggregate_score:.3f}). "
                f"Reverting to pre-optimization agent.{RESET}"
            )
            final_trace = await run_and_trace(
                self._agent.agent_fn, query, context=context,
            )
            final_eval = await self._evaluator.evaluate(final_trace)

        result.final_eval = final_eval
        print(f" done\n")

        final_cp = self._map_checkpoint_results(checkpoints, final_eval, "final")
        result.checkpoint_journey.append(final_cp)

        self._print_comparison_table(baseline_eval, middle_eval, final_eval)

        if checkpoints and len(result.checkpoint_journey) >= 2:
            print(f"\n  {BOLD}Checkpoint journey:{RESET}")
            for idx, cp in enumerate(checkpoints):
                stages: list[str] = []
                for stage_results in result.checkpoint_journey:
                    match = next(
                        (
                            r for r in stage_results
                            if r.checkpoint.checkpoint_id == cp.checkpoint_id
                        ),
                        None,
                    )
                    stages.append(match.status.upper() if match else "--")
                journey_str = " -> ".join(stages)
                print(
                    f"    CP-{idx + 1} {cp.checkpoint_type:<25} {journey_str}"
                )

        if (
            confirmed_metrics
            and optimization_result
            and optimization_result.best_candidate
        ):
            print(f"\n  {BOLD}Prompt efficiency journey:{RESET}")
            sep = f"  +{'-' * 24}+{'-' * 10}+{'-' * 10}+{'-' * 11}+"
            print(sep)
            print(f"  | {'Metric':<22} | {'Before':>8} | {'After':>8} | {'Delta':>9} |")
            print(sep)
            for bp in baseline_prompt_scores:
                name = bp.metric.name
                before = bp.score
                after = optimization_result.final_scores.get(name, before)
                d = after - before
                c = GREEN if d >= 0 else RED
                print(
                    f"  | {name:<22} | {before:>7.1f}% "
                    f"| {after:>7.1f}% | {c}{d:>+8.1f}%{RESET} |"
                )
            print(sep)
            bc = baseline_composite
            ac = optimization_result.best_candidate.composite_score
            dc = ac - bc
            c = GREEN if dc >= 0 else RED
            print(
                f"  | {'COMPOSITE':<22} | {bc:>7.1f}% "
                f"| {ac:>7.1f}% | {c}{dc:>+8.1f}%{RESET} |"
            )
            print(sep)

        total_ms = (time.time() - start) * 1000
        result.total_duration_ms = total_ms

        total_cost = 0.0
        if self._eval_llm:
            total_cost += self._eval_llm.get_cost(self._provider_name or "openai")
        if self._improve_llm:
            total_cost += self._improve_llm.get_cost(self._provider_name or "openai")
        result.total_cost_usd = total_cost

        self._save_results(result, optimization_result, prompts_data)
        print(f"  Duration: {total_ms / 1000:.1f}s | Cost: ${total_cost:.4f}\n")

        return result

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _prompt_user(self, prompt_text: str) -> str:
        """Interactive terminal prompt. Returns 'y', 'n', or the user's text."""
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
        """Return 'y' (auto-approve) when not interactive, else prompt user."""
        if not self._interactive:
            return "y"
        return self._prompt_user(prompt_text)

    def _init_providers(self) -> None:
        """Initialize LLM providers."""
        if self._initialized:
            return
        self._provider_name, self._eval_llm, self._improve_llm = build_providers(
            self.config.llm,
        )
        self._evaluator = TEIEvaluator(self.config, self._eval_llm)
        if self.config.show_cost_estimate:
            print_model_recommendation(self.config.llm)
        self._initialized = True

    def _scan_agent(self) -> tuple[list[str], list[Checkpoint]]:
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
                status = _checkpoint_status(score)
                reasoning = ds.failure_summary or ds.reasoning
            else:
                score = 0.0
                status = "fail"
                reasoning = "Dimension not evaluated"
            results.append(CheckpointResult(
                checkpoint=cp,
                score=score,
                status=status,
                reasoning=reasoning[:300],
                stage=stage,
            ))
        return results

    def _print_batch_summary(
        self,
        batch_number: int,
        batch_log: list[tuple[int, float, float, str]],
        batch_applied: int,
    ) -> None:
        """Print a summary table for a batch of structural fix iterations."""
        sep = f"  +{'-' * 8}+{'-' * 10}+{'-' * 12}+{'-' * 10}+"
        print(f"\n  {BOLD}BATCH {batch_number} SUMMARY ({batch_applied} fixes applied):{RESET}")
        print(sep)
        print(f"  | {'Iter':>6} | {'Score':>8} | {'Delta':>10} | {'Status':<8} |")
        print(sep)
        for iter_num, score, delta, status in batch_log:
            c = GREEN if delta >= 0 else RED
            print(
                f"  | {iter_num:>6} | {score:>8.3f} "
                f"| {c}{delta:>+10.3f}{RESET} | {status:<8} |"
            )
        print(sep)

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
                result.model_dump_json(indent=2), encoding="utf-8",
            )
        except Exception:
            run_path = None

        if opt and opt.best_candidate:
            prompt_path = results_dir / "optimized_prompts.json"
            payload = {
                "original_prompt": prompts_data,
                "optimized_prompt": opt.best_candidate.prompt_text,
                "composite_score": opt.best_candidate.composite_score,
                "metric_scores": opt.best_candidate.metric_scores,
                "iteration": opt.best_candidate.iteration,
            }
            try:
                prompt_path.write_text(
                    json.dumps(payload, indent=2), encoding="utf-8",
                )
            except Exception:
                pass

        saved = run_path or results_dir
        print(f"\n  {GREEN}TEI complete. Results saved to: {saved}{RESET}")

    # ------------------------------------------------------------------ #
    #  Printing                                                            #
    # ------------------------------------------------------------------ #

    def _print_eval_table(self, eval_result: EvalResult, label: str) -> None:
        """Print a formatted evaluation table."""
        sep = f"  +{'-' * 26}+{'-' * 9}+{'-' * 10}+"
        print(f"  {BOLD}{label}:{RESET}")
        print(sep)
        print(f"  | {'Dimension':<24} | {'Score':>7} | {'Status':<8} |")
        print(sep)
        for dim in Dimension:
            ds = eval_result.dimension_scores.get(dim)
            if not ds:
                continue
            status = _score_to_status(ds.score, ds.threshold)
            color = _status_color(status)
            print(
                f"  | {_dim_label(dim):<24} | {ds.score:>5.2f}   "
                f"| {color}{status.upper():<8}{RESET} |"
            )
        print(sep)
        print(
            f"  | {'AGGREGATE':<24} | {eval_result.aggregate_score:>5.2f}   "
            f"| {'':8} |"
        )
        print(sep)

    def _print_checkpoint_details(
        self,
        checkpoints: list[Checkpoint],
        cp_results: list[CheckpointResult],
    ) -> None:
        if not cp_results:
            return
        print(f"\n  {BOLD}Checkpoint details:{RESET}")
        for idx, cr in enumerate(cp_results):
            color = _status_color(cr.status)
            summary = cr.reasoning[:60] if cr.reasoning else ""
            print(
                f"    CP-{idx + 1} {cr.checkpoint.checkpoint_type:<25} "
                f"{color}{cr.status.upper():<6}{RESET} ({summary})"
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
                f"  +{'-' * 26}+{'-' * 10}+{'-' * 10}"
                f"+{'-' * 10}+{'-' * 17}+"
            )
            print(f"  {BOLD}FINAL CHECKMARK REPORT:{RESET}")
            print(sep)
            print(
                f"  | {'Dimension':<24} | {'Baseline':>8} | {'Middle':>8} "
                f"| {'Final':>8} | {'Total Delta':>15} |"
            )
            print(sep)
            for dim in Dimension:
                bds = baseline.dimension_scores.get(dim)
                mds = middle.dimension_scores.get(dim)
                fds = final.dimension_scores.get(dim)
                bs = bds.score if bds else 0.0
                ms = mds.score if mds else bs
                fs = fds.score if fds else ms
                d = fs - bs
                c = GREEN if d >= 0 else RED
                print(
                    f"  | {_dim_label(dim):<24} | {bs:>6.2f}   | {ms:>6.2f}   "
                    f"| {fs:>6.2f}   | {c}{d:>+13.2f}{RESET}   |"
                )
            print(sep)
            ba = baseline.aggregate_score
            ma = middle.aggregate_score
            fa = final.aggregate_score
            td = fa - ba
            c = GREEN if td >= 0 else RED
            print(
                f"  | {'AGGREGATE':<24} | {ba:>6.2f}   | {ma:>6.2f}   "
                f"| {fa:>6.2f}   | {c}{td:>+13.2f}{RESET}   |"
            )
            print(sep)

        elif has_mid or has_fin:
            after = middle if has_mid else final
            stage = "Middle" if has_mid else "Final"
            title = "MIDDLE EVALUATION" if has_mid else "FINAL EVALUATION"
            sep = f"  +{'-' * 26}+{'-' * 10}+{'-' * 10}+{'-' * 11}+"
            print(f"  {BOLD}{title}:{RESET}")
            print(sep)
            print(
                f"  | {'Dimension':<24} | {'Before':>8} "
                f"| {stage:>8} | {'Delta':>9} |"
            )
            print(sep)
            for dim in Dimension:
                bds = baseline.dimension_scores.get(dim)
                ads = after.dimension_scores.get(dim)
                bs = bds.score if bds else 0.0
                asc = ads.score if ads else bs
                d = asc - bs
                c = GREEN if d >= 0 else RED
                print(
                    f"  | {_dim_label(dim):<24} | {bs:>6.2f}   "
                    f"| {asc:>6.2f}   | {c}{d:>+7.2f}{RESET}   |"
                )
            print(sep)
            ba = baseline.aggregate_score
            aa = after.aggregate_score
            d = aa - ba
            c = GREEN if d >= 0 else RED
            print(
                f"  | {'AGGREGATE':<24} | {ba:>6.2f}   "
                f"| {aa:>6.2f}   | {c}{d:>+7.2f}{RESET}   |"
            )
            print(sep)

    def _print_metric_table(
        self,
        results: list[MetricResult],
        composite: float,
        label: str,
    ) -> None:
        sep = f"  +{'-' * 24}+{'-' * 10}+{'-' * 12}+"
        print(f"\n  {BOLD}{label}:{RESET}")
        print(sep)
        print(f"  | {'Metric':<22} | {'Score':>8} | {'Detail':<10} |")
        print(sep)
        for r in results:
            detail = r.detail[:10] if r.detail else ""
            print(
                f"  | {r.metric.name:<22} | {r.score:>7.1f}% | {detail:<10} |"
            )
        print(sep)
        print(
            f"  | {'COMPOSITE (weighted)':<22} | {composite:>7.1f}% | {'':10} |"
        )
        print(sep)

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
            ab = "".join(w[0].upper() for w in m.name.split()[:3])[:4]
            abbrevs.append(ab)

        hdr_parts = [f"{'ID':<6}"]
        for ab in abbrevs:
            hdr_parts.append(f"{ab:>6}")
        hdr_parts.append(f"{'Composite':>10}")
        hdr = " | ".join(hdr_parts)

        rule_parts = ["-" * 6]
        for _ in abbrevs:
            rule_parts.append("-" * 6)
        rule_parts.append("-" * 10)
        rule = "-+-".join(rule_parts)

        print(f"  {BOLD}PARETO FRONT ({len(front)} candidates):{RESET}")
        print(f"  +{rule}+")
        print(f"  | {hdr} |")
        print(f"  +{rule}+")

        best = opt_result.best_candidate
        for cand in front:
            parts = [f"P{cand.iteration:<5}"]
            for mn in metric_names:
                val = cand.metric_scores.get(mn, 0)
                parts.append(f"{val:>5.0f}%")
            parts.append(f"{cand.composite_score:>8.1f}%")
            row = " | ".join(parts)
            tag = ""
            if best and cand.candidate_id == best.candidate_id:
                tag = f"  {BOLD}<-- BEST{RESET}"
            print(f"  | {row} |{tag}")

        print(f"  +{rule}+")

    def _print_optimization_delta(
        self,
        opt_result: OptimizationResult,
        metrics: list[MetricFormula],
    ) -> None:
        sep = f"  +{'-' * 24}+{'-' * 10}+{'-' * 10}+{'-' * 11}+"
        print(f"\n  {BOLD}OPTIMIZATION RESULT:{RESET}")
        print(sep)
        print(
            f"  | {'Metric':<22} | {'Before':>8} | {'After':>8} | {'Delta':>9} |"
        )
        print(sep)

        w_before = 0.0
        w_after = 0.0
        for m in metrics:
            before = opt_result.baseline_scores.get(m.name, 0)
            after = opt_result.final_scores.get(m.name, 0)
            d = after - before
            c = GREEN if d >= 0 else RED
            w_before += before * m.weight
            w_after += after * m.weight
            print(
                f"  | {m.name:<22} | {before:>7.0f}% "
                f"| {after:>7.0f}% | {c}{d:>+8.0f}%{RESET} |"
            )

        print(sep)
        tw = sum(m.weight for m in metrics) or 1.0
        cb = w_before / tw
        ca = w_after / tw
        cd = ca - cb
        c = GREEN if cd >= 0 else RED
        print(
            f"  | {'COMPOSITE':<22} | {cb:>7.1f}% "
            f"| {ca:>7.1f}% | {c}{cd:>+8.1f}%{RESET} |"
        )
        print(sep)

    # ------------------------------------------------------------------ #
    #  Legacy methods (kept for backward compatibility)                     #
    # ------------------------------------------------------------------ #

    async def evaluate_only(
        self,
        query: Any,
        context: Optional[dict[str, Any]] = None,
    ) -> EvalResult:
        """[Legacy] Run the agent once and evaluate without improvement."""
        self._init_providers()
        trace = await run_and_trace(self._agent.agent_fn, query, context=context)
        return await self._evaluator.evaluate(trace)

    async def compare(
        self,
        query: Any,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, EvalResult]:
        """[Legacy] Run baseline and full TEI loop, return both for comparison."""
        baseline = await self.evaluate_only(query, context)
        full = await self.run(query, context=context)
        return {
            "baseline": baseline,
            "final": full.final_eval,
        }
