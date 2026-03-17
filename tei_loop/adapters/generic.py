"""
Generic Python Adapter.

Wraps any Python callable (sync or async) as a TEI-compatible agent.
This is the primary adapter and covers 100% of agents since every
framework (LangGraph, CrewAI, custom) produces a Python callable.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from ..tracer import TEITracer, run_and_trace
from ..models import Trace


class GenericAdapter:
    """Wraps any Python callable as a TEI agent."""

    def __init__(
        self,
        agent_fn: Callable,
        name: str = "agent",
        description: str = "",
    ):
        self.agent_fn = agent_fn
        self.name = name or getattr(agent_fn, "__name__", "agent")
        self.description = description
        self._tracer = TEITracer()

    async def run(self, query: Any, context: Optional[dict[str, Any]] = None) -> Trace:
        return await run_and_trace(
            self.agent_fn,
            query,
            tracer=self._tracer,
            context=context,
        )

    def __repr__(self) -> str:
        return f"GenericAdapter(name={self.name!r})"
