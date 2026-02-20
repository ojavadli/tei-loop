"""Tests for TEI data models."""

import pytest
from tei_loop.models import (
    Assertion,
    AssertionVerdict,
    Checkpoint,
    CheckpointResult,
    Dimension,
    DimensionScore,
    EvalResult,
    Failure,
    Fix,
    FixStrategyType,
    IterationResult,
    MetricFormula,
    MetricResult,
    OptimizationResult,
    ParetoCandidate,
    RunMode,
    Severity,
    StructuralFix,
    TEIConfig,
    TEIFullResult,
    TEIResult,
    Trace,
    TraceStep,
)


def test_dimension_enum():
    assert len(Dimension) == 4
    assert Dimension.TARGET_ALIGNMENT.value == "target_alignment"
    assert Dimension.REASONING_SOUNDNESS.value == "reasoning_soundness"
    assert Dimension.EXECUTION_ACCURACY.value == "execution_accuracy"
    assert Dimension.OUTPUT_INTEGRITY.value == "output_integrity"


def test_trace_step_defaults():
    step = TraceStep(name="test")
    assert step.step_type == "generic"
    assert step.step_id
    assert step.timestamp > 0


def test_trace_creation():
    trace = Trace(
        agent_input="What is 2+2?",
        agent_output="4",
        steps=[TraceStep(name="compute")],
    )
    assert trace.trace_id
    assert trace.agent_input == "What is 2+2?"
    assert len(trace.steps) == 1


def test_dimension_score_never_100():
    ds = DimensionScore(dimension=Dimension.TARGET_ALIGNMENT, score=0.97)
    assert ds.score <= 0.97


def test_eval_result_summary():
    er = EvalResult(
        dimension_scores={
            Dimension.TARGET_ALIGNMENT: DimensionScore(
                dimension=Dimension.TARGET_ALIGNMENT, score=0.85, passed=True, threshold=0.7
            ),
            Dimension.OUTPUT_INTEGRITY: DimensionScore(
                dimension=Dimension.OUTPUT_INTEGRITY, score=0.55, passed=False, threshold=0.7
            ),
        },
        aggregate_score=0.70,
        all_passed=False,
    )
    assert not er.all_passed
    failures = er.get_failures()
    assert len(failures) == 1
    assert failures[0].dimension == Dimension.OUTPUT_INTEGRITY


def test_failure_creation():
    f = Failure(
        dimension=Dimension.REASONING_SOUNDNESS,
        severity=Severity.MAJOR,
        description="Contradictory reasoning about protein content",
        suggested_strategy=FixStrategyType.REASONING_REGENERATE,
    )
    assert f.failure_id
    assert f.severity == Severity.MAJOR


def test_fix_creation():
    fix = Fix(
        strategy=FixStrategyType.TARGET_REANCHOR,
        replacement_content="Focus on the original question about dessert preferences",
        rationale="Agent drifted to discussing general nutrition",
    )
    assert fix.fix_id
    assert not fix.applied


def test_tei_result_before_after():
    r = TEIResult(
        iterations=[
            IterationResult(
                iteration=0,
                eval_result=EvalResult(
                    dimension_scores={
                        Dimension.TARGET_ALIGNMENT: DimensionScore(
                            dimension=Dimension.TARGET_ALIGNMENT, score=0.55, threshold=0.7
                        ),
                    },
                    aggregate_score=0.55,
                ),
            ),
            IterationResult(
                iteration=1,
                eval_result=EvalResult(
                    dimension_scores={
                        Dimension.TARGET_ALIGNMENT: DimensionScore(
                            dimension=Dimension.TARGET_ALIGNMENT, score=0.82, threshold=0.7
                        ),
                    },
                    aggregate_score=0.82,
                ),
            ),
        ],
        baseline_score=0.55,
        final_score=0.82,
        improvement_delta=0.27,
    )
    assert r.before_scores["target_alignment"] == 0.55
    assert r.after_scores["target_alignment"] == 0.82
    assert "0.55" in r.summary()


def test_config_defaults():
    config = TEIConfig()
    assert config.mode == RunMode.RUNTIME
    assert config.max_retries == 3
    assert config.parallel_eval is True
    assert len(config.dimensions) == 4
    assert config.dimensions[Dimension.TARGET_ALIGNMENT].threshold == 0.7


def test_checkpoint_instantiation():
    cp = Checkpoint(
        file_path="/path/to/agent.py",
        line_number=42,
        dimension=Dimension.OUTPUT_INTEGRITY,
    )
    assert cp.checkpoint_id
    assert cp.file_path == "/path/to/agent.py"
    assert cp.line_number == 42
    assert cp.dimension == Dimension.OUTPUT_INTEGRITY


def test_checkpoint_result_instantiation():
    cp = Checkpoint(
        file_path="/path/to/agent.py",
        line_number=42,
        dimension=Dimension.OUTPUT_INTEGRITY,
    )
    cr = CheckpointResult(checkpoint=cp, score=0.85, status="pass")
    assert cr.checkpoint is cp
    assert cr.score == 0.85
    assert cr.status == "pass"


def test_structural_fix_instantiation():
    sf = StructuralFix(
        checkpoint_id="cp123",
        file_path="/path/to/agent.py",
        issue="Missing validation",
        proposed_fix="Add input validation",
    )
    assert sf.fix_id
    assert sf.checkpoint_id == "cp123"
    assert sf.issue == "Missing validation"
    assert not sf.applied


def test_metric_formula_instantiation():
    mf = MetricFormula(
        name="topic_coverage",
        description="Covers all required topics",
        formula="count(matched_topics) / count(required_topics)",
    )
    assert mf.metric_id
    assert mf.name == "topic_coverage"
    assert mf.weight == 0.25


def test_metric_result_instantiation():
    mf = MetricFormula(
        name="topic_coverage",
        description="Covers all required topics",
        formula="count(matched_topics) / count(required_topics)",
    )
    mr = MetricResult(metric=mf, score=75.0, detail="8/12")
    assert mr.metric is mf
    assert mr.score == 75.0
    assert mr.detail == "8/12"


def test_pareto_candidate_instantiation():
    pc = ParetoCandidate(
        iteration=1,
        prompt_text="You are a helpful assistant.",
        metric_scores={"m1": 80.0, "m2": 60.0},
        composite_score=70.0,
    )
    assert pc.candidate_id
    assert pc.iteration == 1
    assert pc.prompt_text == "You are a helpful assistant."
    assert pc.metric_scores["m1"] == 80.0


def test_optimization_result_instantiation():
    pc = ParetoCandidate(iteration=0, prompt_text="test", composite_score=70.0)
    opt = OptimizationResult(
        total_iterations=10,
        pareto_front=[pc],
        best_candidate=pc,
    )
    assert opt.total_iterations == 10
    assert len(opt.pareto_front) == 1
    assert opt.best_candidate is pc


def test_tei_full_result_instantiation():
    result = TEIFullResult()
    assert result.run_id
    assert result.agent_files == []
    assert result.checkpoints == []
    assert result.optimization is None
