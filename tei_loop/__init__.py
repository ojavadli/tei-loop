"""
TEI Loop: Target -> Evaluate -> Improve

A self-improving loop for agentic systems with iterative optimization.

Usage:
    from tei_loop import TEILoop

    loop = TEILoop(agent=my_agent_function, agent_file="my_agent.py")
    result = await loop.run("user query", test_queries=["q1", "q2", "q3"])
    print(result)
"""

from .models import (
    Dimension,
    TEIConfig,
    TEIResult,
    TEIFullResult,
    EvalResult,
    Trace,
    TraceStep,
    Failure,
    Fix,
    RunMode,
    Checkpoint,
    CheckpointResult,
    StructuralFix,
    MetricFormula,
    MetricResult,
    ParetoCandidate,
    OptimizationResult,
)
from .loop import TEILoop
from .tracer import tei_trace
from .evaluator import TEIEvaluator
from .checkpoint_scanner import scan_agent
from .pareto import update_pareto_front, select_best

__version__ = "1.0.0"

__all__ = [
    "TEILoop",
    "TEIConfig",
    "TEIResult",
    "TEIFullResult",
    "EvalResult",
    "Trace",
    "TraceStep",
    "Dimension",
    "Failure",
    "Fix",
    "RunMode",
    "TEIEvaluator",
    "tei_trace",
    "scan_agent",
    "Checkpoint",
    "CheckpointResult",
    "StructuralFix",
    "MetricFormula",
    "MetricResult",
    "ParetoCandidate",
    "OptimizationResult",
    "update_pareto_front",
    "select_best",
]
