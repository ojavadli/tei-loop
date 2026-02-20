"""
TEI Loop data models.

All structured types used across tracer, evaluator, improver, and loop controller.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Dimension(str, Enum):
    """The four TEI evaluation dimensions."""
    TARGET_ALIGNMENT = "target_alignment"
    REASONING_SOUNDNESS = "reasoning_soundness"
    EXECUTION_ACCURACY = "execution_accuracy"
    OUTPUT_INTEGRITY = "output_integrity"


class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class FixStrategyType(str, Enum):
    TARGET_REANCHOR = "target_reanchor"
    REASONING_REGENERATE = "reasoning_regenerate"
    EXECUTION_CORRECT = "execution_correct"
    OUTPUT_REPAIR = "output_repair"


class RunMode(str, Enum):
    RUNTIME = "runtime"
    DEVELOPMENT = "development"


class AssertionVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"


class TraceStep(BaseModel):
    """A single step in an agent execution trace."""
    step_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    step_type: str = "generic"
    name: str = ""
    input_data: Any = None
    output_data: Any = None
    duration_ms: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class Trace(BaseModel):
    """Complete execution trace for one agent run."""
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    agent_input: Any = None
    agent_output: Any = None
    steps: list[TraceStep] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Assertion(BaseModel):
    """A verifiable claim extracted from the agent output and checked against evidence."""
    assertion_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    claim: str
    evidence: str = ""
    verdict: AssertionVerdict = AssertionVerdict.FAIL
    dimension: Dimension = Dimension.OUTPUT_INTEGRITY
    source_step_id: Optional[str] = None
    explanation: str = ""


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""
    dimension: Dimension
    score: float = Field(ge=0.0, le=0.97)
    passed: bool = False
    threshold: float = 0.7
    assertions: list[Assertion] = Field(default_factory=list)
    reasoning: str = ""
    failure_summary: Optional[str] = None


class EvalResult(BaseModel):
    """Complete evaluation result across all four TEI dimensions."""
    eval_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    trace_id: str = ""
    dimension_scores: dict[Dimension, DimensionScore] = Field(default_factory=dict)
    aggregate_score: float = 0.0
    all_passed: bool = False
    total_assertions: int = 0
    passed_assertions: int = 0
    failed_assertions: int = 0
    eval_duration_ms: float = 0.0
    eval_cost_usd: float = 0.0
    timestamp: float = Field(default_factory=time.time)

    def get_failures(self) -> list[DimensionScore]:
        return [ds for ds in self.dimension_scores.values() if not ds.passed]

    def summary(self) -> str:
        lines = [f"TEI Evaluation  Aggregate: {self.aggregate_score:.2f}"]
        for dim in Dimension:
            ds = self.dimension_scores.get(dim)
            if ds:
                status = "PASS" if ds.passed else "FAIL"
                lines.append(
                    f"  {dim.value}: {ds.score:.2f} [{status}] (threshold {ds.threshold})"
                )
        lines.append(f"  Assertions: {self.passed_assertions}/{self.total_assertions} passed")
        return "\n".join(lines)


class Failure(BaseModel):
    """A diagnosed failure from evaluation."""
    failure_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    dimension: Dimension
    severity: Severity
    description: str
    source_step_id: Optional[str] = None
    evidence: str = ""
    suggested_strategy: FixStrategyType = FixStrategyType.OUTPUT_REPAIR


class Fix(BaseModel):
    """A concrete fix to apply to the agent next run."""
    fix_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    strategy: FixStrategyType
    failure_id: str = ""
    original_content: str = ""
    replacement_content: str = ""
    rationale: str = ""
    applied: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class IterationResult(BaseModel):
    """Result of a single TEI iteration (one evaluate-improve cycle)."""
    iteration: int = 0
    trace: Trace = Field(default_factory=Trace)
    eval_result: EvalResult = Field(default_factory=EvalResult)
    failures: list[Failure] = Field(default_factory=list)
    fixes: list[Fix] = Field(default_factory=list)
    improved: bool = False


class TEIResult(BaseModel):
    """Final result of a complete TEI loop run."""
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    mode: RunMode = RunMode.RUNTIME
    query: str = ""
    final_output: Any = None
    iterations: list[IterationResult] = Field(default_factory=list)
    baseline_score: float = 0.0
    final_score: float = 0.0
    improvement_delta: float = 0.0
    total_iterations: int = 0
    converged: bool = False
    total_duration_ms: float = 0.0
    total_cost_usd: float = 0.0
    timestamp: float = Field(default_factory=time.time)

    @property
    def before_scores(self) -> dict[str, float]:
        if not self.iterations:
            return {}
        first = self.iterations[0].eval_result
        return {
            d.value: first.dimension_scores[d].score
            for d in Dimension
            if d in first.dimension_scores
        }

    @property
    def after_scores(self) -> dict[str, float]:
        if not self.iterations:
            return {}
        best = max(self.iterations, key=lambda it: it.eval_result.aggregate_score)
        return {
            d.value: best.eval_result.dimension_scores[d].score
            for d in Dimension
            if d in best.eval_result.dimension_scores
        }

    def summary(self) -> str:
        lines = [
            f"TEI Result  {self.mode.value} mode",
            f"  Query: {str(self.query)[:80]}",
            f"  Iterations: {self.total_iterations}",
            f"  Baseline score: {self.baseline_score:.2f}",
            f"  Final score:    {self.final_score:.2f}",
            f"  Improvement:    {self.improvement_delta:+.2f}",
            f"  Converged: {self.converged}",
            f"  Duration: {self.total_duration_ms:.0f}ms",
            f"  Cost: ${self.total_cost_usd:.4f}",
            "",
            "  Before -> After per dimension:",
        ]
        for dim_name in self.before_scores:
            before = self.before_scores.get(dim_name, 0)
            after = self.after_scores.get(dim_name, 0)
            delta = after - before
            lines.append(f"    {dim_name}: {before:.2f} -> {after:.2f} ({delta:+.2f})")
        return "\n".join(lines)


class DimensionConfig(BaseModel):
    """Configuration for a single evaluation dimension."""
    enabled: bool = True
    threshold: float = 0.7
    weight: float = 1.0


class LLMConfig(BaseModel):
    """LLM configuration for TEI internal evaluation and improvement calls."""
    provider: str = "auto"
    eval_model: str = "auto"
    improve_model: str = "auto"
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4096


class TEIConfig(BaseModel):
    """Top-level TEI configuration."""
    mode: RunMode = RunMode.RUNTIME
    max_retries: int = 3
    max_dev_iterations: int = 50
    convergence_threshold: float = 0.02
    parallel_eval: bool = True
    llm: LLMConfig = Field(default_factory=LLMConfig)
    dimensions: dict[Dimension, DimensionConfig] = Field(
        default_factory=lambda: {
            Dimension.TARGET_ALIGNMENT: DimensionConfig(threshold=0.7, weight=1.0),
            Dimension.REASONING_SOUNDNESS: DimensionConfig(threshold=0.65, weight=1.0),
            Dimension.EXECUTION_ACCURACY: DimensionConfig(threshold=0.7, weight=1.0),
            Dimension.OUTPUT_INTEGRITY: DimensionConfig(threshold=0.7, weight=1.0),
        }
    )
    show_cost_estimate: bool = True
    verbose: bool = False


class Checkpoint(BaseModel):
    """A checkpoint placed by the scanner."""
    checkpoint_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    file_path: str
    line_number: int
    code_snippet: str = ""
    checkpoint_type: str = ""  # e.g. "llm_call", "tool_call", "db_write", "output_render", "prompt_injection"
    dimension: Dimension
    description: str = ""
    status: str = "pending"  # pending/pass/weak/fail


class CheckpointResult(BaseModel):
    """Result of evaluating a checkpoint."""
    checkpoint: Checkpoint
    score: float = Field(ge=0.0, le=1.0)
    status: str = ""  # pass/weak/fail
    reasoning: str = ""
    stage: str = "baseline"  # baseline/middle/final


class StructuralFix(BaseModel):
    """A proposed structural fix."""
    fix_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    checkpoint_id: str
    file_path: str
    issue: str
    proposed_fix: str
    expected_impact: str = ""
    approved: bool = False
    user_alternative: Optional[str] = None
    applied: bool = False
    code_patch: str = ""


class MetricFormula(BaseModel):
    """An objective metric with formula."""
    metric_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    name: str
    description: str
    formula: str  # human-readable formula string
    measurement_method: str = "llm_judge"  # llm_judge/code_based/hybrid
    weight: float = 0.25
    approved: bool = False


class MetricResult(BaseModel):
    """Result of measuring a metric."""
    metric: MetricFormula
    score: float = Field(ge=0.0, le=100.0)
    detail: str = ""  # e.g. "8/12" for topic coverage
    raw_data: dict[str, Any] = Field(default_factory=dict)


class ParetoCandidate(BaseModel):
    """A candidate in the Pareto front."""
    candidate_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    iteration: int
    prompt_text: str
    metric_scores: dict[str, float] = Field(default_factory=dict)
    composite_score: float = 0.0
    parent_ids: list[str] = Field(default_factory=list)
    strategy: str = "mutation"  # mutation/merge/baseline
    reflection: str = ""
    dominated: bool = False


class OptimizationResult(BaseModel):
    """Final result of prompt optimization."""
    total_iterations: int
    pareto_front: list[ParetoCandidate] = Field(default_factory=list)
    best_candidate: Optional[ParetoCandidate] = None
    metric_history: list[dict[str, float]] = Field(default_factory=list)
    baseline_scores: dict[str, float] = Field(default_factory=dict)
    final_scores: dict[str, float] = Field(default_factory=dict)


class TEIFullResult(BaseModel):
    """Final result of the entire 8-step TEI run."""
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    agent_files: list[str] = Field(default_factory=list)
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    baseline_eval: Optional[EvalResult] = None
    structural_fixes: list[StructuralFix] = Field(default_factory=list)
    middle_eval: Optional[EvalResult] = None
    metrics: list[MetricFormula] = Field(default_factory=list)
    baseline_prompt_scores: list[MetricResult] = Field(default_factory=list)
    optimization: Optional[OptimizationResult] = None
    final_eval: Optional[EvalResult] = None
    checkpoint_journey: list[list[CheckpointResult]] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    total_cost_usd: float = 0.0
    timestamp: float = Field(default_factory=time.time)
