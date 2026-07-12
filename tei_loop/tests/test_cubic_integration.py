"""Offline integration tests for D-ARC modes in PromptOptimizer + CLI + models.

Covers spec categories 12-16 and 20:
  12 all evaluated candidates still reach the Pareto archive
  13 pareto mode preserves previous behavior
  14 hybrid final selection and baseline fallback
  15 backward-compatible OptimizationResult deserialization
  16 CLI parsing for all new settings
  20 fake end-to-end optimizer execution with no network

No test here touches the network: the provider is mocked, the patched agent
is replaced by a deterministic local function, and a socket guard enforces it.
"""
import asyncio
import json
import re
import socket
import subprocess
import sys

import pytest

from tei_loop.cubic import CubicConfig
from tei_loop.models import MetricFormula, OptimizationResult
from tei_loop.llm_provider import BaseLLMProvider
from tei_loop.prompt_optimizer import PromptOptimizer


# ---------------------------------------------------------------------------
# no-network fakes
# ---------------------------------------------------------------------------

class NoNetwork:
    """Context manager: any outbound network attempt fails the test.

    Blocks name resolution and TCP connection establishment rather than
    socket.socket itself, because asyncio's event loop legitimately creates
    an internal (non-network) self-pipe socket pair."""
    def __enter__(self):
        self._saved_gai = socket.getaddrinfo
        self._saved_cc = socket.create_connection

        def _blocked(*a, **k):
            raise AssertionError("network access attempted during offline test")
        socket.getaddrinfo = _blocked
        socket.create_connection = _blocked
        return self

    def __exit__(self, *exc):
        socket.getaddrinfo = self._saved_gai
        socket.create_connection = self._saved_cc
        return False


class FakeProvider(BaseLLMProvider):
    """Deterministic offline provider.

    generate():       used by PromptEvaluator — reads the metric name from the
                      user prompt and the candidate's quality marker
                      "[[Q:m1=80,m2=40]]" from the echoed agent output, and
                      returns that score as judge JSON.
    generate_json():  used by the optimizer — returns the next scripted batch
                      of proposals (for cubic/hybrid) or a scripted mutation
                      (for pareto, via generate()).
    """

    def __init__(self, proposal_batches=None):
        super().__init__(api_key="fake", model="fake-model")
        self.proposal_batches = list(proposal_batches or [])
        self.json_calls = 0

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        m_metric = re.search(r"\*\*Name:\*\* (\w+)", user_prompt)
        m_marker = re.search(r"\[\[Q:([^\]]+)\]\]", user_prompt)
        if m_metric and m_marker:
            metric = m_metric.group(1)
            scores = dict(kv.split("=") for kv in m_marker.group(1).split(","))
            score = float(scores.get(metric, 0))
            return json.dumps({"score": score, "detail": "scripted",
                               "reasoning": "offline fake"})
        # pareto-mode mutation/merge path: return a new prompt variant
        return "Improved prompt variant [[Q:m1=60,m2=60]] with enough length."

    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        if self.proposal_batches:
            batch = self.proposal_batches[min(self.json_calls,
                                              len(self.proposal_batches) - 1)]
            self.json_calls += 1
            return {"proposals": batch}
        self.json_calls += 1
        return {"proposals": []}


def _metric(name: str, weight: float = 0.5) -> MetricFormula:
    return MetricFormula(name=name, description=f"metric {name}",
                         formula=f"{name} formula", weight=weight,
                         measurement_method="llm_judge")


def _local_agent_factory(optimizer: PromptOptimizer):
    """Replace network-calling patched agents with a deterministic echo:
    the agent output carries the prompt's quality marker so the fake judge
    can score it."""
    def _create(prompt_text: str):
        marker = re.search(r"\[\[Q:[^\]]+\]\]", prompt_text)
        tag = marker.group(0) if marker else "[[Q:m1=50,m2=50]]"

        def agent(query: str) -> str:
            return f"answer to {query} {tag}"
        return agent
    optimizer._create_patched_agent = _create
    return optimizer


def _proposal(m1: int, m2: int, scale: str, tag: str) -> dict:
    return {
        "prompt": (f"Candidate {tag} [[Q:m1={m1},m2={m2}]] — a sufficiently "
                   f"long prompt body for validity checks."),
        "requested_scale": scale,
        "rationale": f"scripted {tag}",
    }


def _mk_optimizer(mode: str, batches, warmup=1, patience=3) -> PromptOptimizer:
    opt = PromptOptimizer(
        improve_llm=FakeProvider(proposal_batches=batches),
        eval_llm=FakeProvider(),
        metrics=[_metric("m1"), _metric("m2")],
        agent_fn=lambda q: f"base {q} [[Q:m1=50,m2=50]]",
        agent_file=None,
        optimizer_mode=mode,
        cubic_config=CubicConfig(warmup_evals=warmup, patience=patience),
    )
    return _local_agent_factory(opt)


BASE_PROMPT = "Baseline prompt [[Q:m1=50,m2=50]] with plenty of length to pass."
QUERIES = ["q1", "q2", "q3"]


# ---------------------------------------------------------------------------
# 20 + 12: fake end-to-end, archive coexistence
# ---------------------------------------------------------------------------

class TestEndToEndOffline:
    def test_cubic_end_to_end_no_network(self):
        batches = [
            [_proposal(70, 60, "micro", "A")],       # warmup: improvement
            [_proposal(40, 90, "small", "B")],       # tradeoff: U tie-ish
            [_proposal(80, 75, "medium", "C")],      # clear improvement
            [_proposal(30, 30, "large", "D")],       # bad
        ]
        opt = _mk_optimizer("cubic", batches, warmup=1)
        with NoNetwork():
            res = asyncio.run(
                opt.optimize(BASE_PROMPT, QUERIES, num_iterations=4, verbose=False))
        assert res.optimizer_mode == "cubic"
        assert res.final_selection_source == "cubic_best_accepted_incumbent"
        assert res.cubic_config and res.cubic_config["warmup_evals"] == 1
        assert len(res.cubic_history) >= 3
        # sigma/rho/pred diagnostics serialized on every non-warmup record
        for rec in res.cubic_history:
            assert "sigma_before" in rec and "sigma_after" in rec
            if rec["phase"] == "cubic":
                assert rec["rho"] is None or isinstance(rec["rho"], float)
        # JSON-safe (no Infinity)
        blob = json.dumps(res.model_dump(mode="json"))
        assert "Infinity" not in blob

    def test_all_evaluated_candidates_reach_archive(self):
        # A and B are metric tradeoffs: both non-dominated regardless of
        # D-ARC acceptance; both must be in the final Pareto archive.
        batches = [
            [_proposal(90, 20, "micro", "A")],
            [_proposal(20, 90, "small", "B")],
        ]
        opt = _mk_optimizer("hybrid", batches, warmup=2)
        with NoNetwork():
            res = asyncio.run(
                opt.optimize(BASE_PROMPT, QUERIES, num_iterations=2, verbose=False))
        scores = sorted((c.metric_scores["m1"], c.metric_scores["m2"])
                        for c in res.pareto_front)
        assert (20.0, 90.0) in scores and (90.0, 20.0) in scores

    def test_incumbent_and_surrogate_histories_persisted(self):
        batches = [[_proposal(75, 70, "micro", "A")],
                   [_proposal(78, 72, "small", "B")],
                   [_proposal(60, 55, "medium", "C")]]
        opt = _mk_optimizer("cubic", batches, warmup=1)
        with NoNetwork():
            res = asyncio.run(
                opt.optimize(BASE_PROMPT, QUERIES, num_iterations=3, verbose=False))
        assert res.incumbent_history[0]["event"] == "baseline"
        assert any(e["event"] != "baseline" for e in res.incumbent_history)
        assert res.scalar_utility_baseline == pytest.approx(0.5, abs=1e-6)
        assert res.scalar_utility_final is not None


# ---------------------------------------------------------------------------
# 13: pareto mode unchanged
# ---------------------------------------------------------------------------

class TestParetoModePreserved:
    def test_pareto_default_and_behavior(self):
        opt = PromptOptimizer(
            improve_llm=FakeProvider(),
            eval_llm=FakeProvider(),
            metrics=[_metric("m1"), _metric("m2")],
            agent_fn=lambda q: f"base {q} [[Q:m1=50,m2=50]]",
        )
        assert opt.optimizer_mode == "pareto"       # default unchanged
        _local_agent_factory(opt)
        with NoNetwork():
            res = asyncio.run(
                opt.optimize(BASE_PROMPT, QUERIES, num_iterations=3, verbose=False))
        assert res.optimizer_mode == "pareto"
        assert res.final_selection_source == "pareto_composite"
        assert res.total_iterations == 3
        assert res.cubic_history == []              # no D-ARC state leaks in
        assert res.best_candidate is not None
        assert res.pareto_front

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="optimizer_mode"):
            PromptOptimizer(
                improve_llm=FakeProvider(), eval_llm=FakeProvider(),
                metrics=[_metric("m1")], agent_fn=lambda q: q,
                optimizer_mode="nsga2")


# ---------------------------------------------------------------------------
# 14: hybrid final selection + baseline fallback
# ---------------------------------------------------------------------------

class TestHybridSelection:
    def test_hybrid_ranks_by_scalar_utility(self):
        batches = [
            [_proposal(90, 20, "micro", "A")],   # U = .55, non-dominated
            [_proposal(70, 80, "small", "B")],   # U = .75, non-dominated
        ]
        opt = _mk_optimizer("hybrid", batches, warmup=2)
        with NoNetwork():
            res = asyncio.run(
                opt.optimize(BASE_PROMPT, QUERIES, num_iterations=2, verbose=False))
        assert res.final_selection_source == "hybrid_pareto_scalar_utility_rank"
        assert res.best_candidate.metric_scores == {"m1": 70.0, "m2": 80.0}
        assert res.scalar_utility_final == pytest.approx(0.75, abs=1e-6)

    def test_hybrid_baseline_fallback_when_all_worse(self):
        batches = [
            [_proposal(40, 40, "micro", "A")],
            [_proposal(30, 20, "small", "B")],
        ]
        opt = _mk_optimizer("hybrid", batches, warmup=2)
        with NoNetwork():
            res = asyncio.run(
                opt.optimize(BASE_PROMPT, QUERIES, num_iterations=2, verbose=False))
        # every candidate is worse: the scalar-utility argmax over the archive
        # is the baseline itself -> baseline prompt is the final choice
        assert res.best_candidate.strategy == "baseline"
        assert res.best_candidate.prompt_text == BASE_PROMPT
        assert res.scalar_utility_final == pytest.approx(
            res.scalar_utility_baseline, abs=1e-9)


# ---------------------------------------------------------------------------
# 15: backward-compatible deserialization
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_old_serialized_result_loads(self):
        legacy = {
            "total_iterations": 30,
            "pareto_front": [{
                "candidate_id": "abc123", "iteration": 3,
                "prompt_text": "old prompt", "metric_scores": {"m1": 80.0},
                "composite_score": 80.0, "parent_ids": [], "strategy": "mutation",
                "reflection": "", "dominated": False,
            }],
            "best_candidate": None,
            "metric_history": [{"m1": 70.0}],
            "baseline_scores": {"m1": 70.0},
            "final_scores": {"m1": 80.0},
        }
        res = OptimizationResult.model_validate(legacy)
        assert res.optimizer_mode == "pareto"            # default applied
        assert res.cubic_config is None
        assert res.cubic_history == []
        assert res.final_selection_source == "pareto_composite"
        # round-trips including the new fields
        again = OptimizationResult.model_validate(
            json.loads(res.model_dump_json()))
        assert again.total_iterations == 30


# ---------------------------------------------------------------------------
# 16: CLI parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:
    def _run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "tei_loop", *args],
            capture_output=True, text=True, timeout=120,
            cwd="/tmp",
        )

    def test_new_flags_accepted(self):
        # file-not-found exits 1 AFTER argparse accepted every new flag
        res = self._run_cli(
            "definitely_missing_agent_file.py",
            "--optimizer-mode", "cubic", "--cubic-sigma", "2.0",
            "--cubic-warmup", "3", "--cubic-proposals", "4",
            "--cubic-window", "10", "--cubic-patience", "2",
        )
        assert res.returncode == 1
        assert "usage" not in res.stderr.lower()

    def test_invalid_mode_rejected_by_argparse(self):
        res = self._run_cli("x.py", "--optimizer-mode", "banana")
        assert res.returncode == 2
        assert "--optimizer-mode" in res.stderr
