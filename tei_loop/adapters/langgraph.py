"""
LangGraph Adapter — deep node-level tracing.

Hooks into LangGraph's compiled graph to capture each node execution
as a separate TraceStep, giving TEI per-node visibility for diagnosis.

Works with LangGraph >=0.2 compiled StateGraphs.
If LangGraph is not installed, falls back to generic black-box tracing.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Callable, Optional

from ..tracer import TEITracer, _safe_serialize
from ..models import Trace, TraceStep


class LangGraphAdapter:
    """Wraps a LangGraph compiled graph for TEI with per-node tracing."""

    def __init__(self, graph: Any, name: str = "langgraph_agent"):
        self.graph = graph
        self.name = name
        self._tracer = TEITracer()

    async def run(self, query: Any, context: Optional[dict[str, Any]] = None) -> Trace:
        self._tracer.start()
        overall_start = time.time()

        input_payload = self._build_input(query)
        output = None
        error_msg = None

        try:
            if self._has_stream_events():
                output = await self._run_with_stream(input_payload)
            elif hasattr(self.graph, "ainvoke"):
                output = await self._run_async(input_payload)
            else:
                output = await self._run_sync(input_payload)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"

        overall_ms = (time.time() - overall_start) * 1000
        self._tracer.add_step(
            name="langgraph_total",
            step_type="orchestrator",
            input_data=query,
            output_data=output,
            duration_ms=overall_ms,
            error=error_msg,
            metadata={"adapter": "langgraph", "node_count": len(self._tracer._steps)},
        )

        return self._tracer.finish(agent_input=query, agent_output=output)

    def _build_input(self, query: Any) -> dict[str, Any]:
        if isinstance(query, dict):
            return query
        return {"messages": [("user", str(query))]}

    def _has_stream_events(self) -> bool:
        return hasattr(self.graph, "astream_events")

    async def _run_with_stream(self, input_payload: dict[str, Any]) -> Any:
        """Use LangGraph's astream_events to capture per-node execution."""
        final_output = None
        current_node: Optional[str] = None
        node_start: float = 0.0
        node_input: Any = None

        async for event in self.graph.astream_events(input_payload, version="v2"):
            kind = event.get("event", "")
            name = event.get("name", "")
            data = event.get("data", {})

            if kind == "on_chain_start" and name != self.name:
                current_node = name
                node_start = time.time()
                node_input = data.get("input")

            elif kind == "on_chain_end" and name == current_node:
                node_output = data.get("output")
                node_ms = (time.time() - node_start) * 1000

                self._tracer.add_step(
                    name=current_node,
                    step_type="langgraph_node",
                    input_data=node_input,
                    output_data=node_output,
                    duration_ms=node_ms,
                    metadata={"event_kind": kind},
                )
                final_output = node_output
                current_node = None

            elif kind == "on_chain_error" and name == current_node:
                node_ms = (time.time() - node_start) * 1000
                self._tracer.add_step(
                    name=current_node,
                    step_type="langgraph_node",
                    input_data=node_input,
                    duration_ms=node_ms,
                    error=str(data.get("error", "unknown")),
                )
                current_node = None

            elif kind == "on_tool_start":
                self._tracer.add_step(
                    name=name,
                    step_type="tool_call",
                    input_data=data.get("input"),
                    metadata={"tool_name": name},
                )

            elif kind == "on_tool_end":
                if self._tracer._steps and self._tracer._steps[-1].name == name:
                    self._tracer._steps[-1].output_data = _safe_serialize(
                        data.get("output")
                    )

        return final_output

    async def _run_async(self, input_payload: dict[str, Any]) -> Any:
        """Fallback: ainvoke without streaming (less granular)."""
        node_start = time.time()
        result = await self.graph.ainvoke(input_payload)
        node_ms = (time.time() - node_start) * 1000

        self._tracer.add_step(
            name="ainvoke",
            step_type="langgraph_invoke",
            input_data=input_payload,
            output_data=result,
            duration_ms=node_ms,
        )
        return result

    async def _run_sync(self, input_payload: dict[str, Any]) -> Any:
        """Fallback: sync invoke wrapped in thread."""
        node_start = time.time()
        result = await asyncio.to_thread(self.graph.invoke, input_payload)
        node_ms = (time.time() - node_start) * 1000

        self._tracer.add_step(
            name="invoke",
            step_type="langgraph_invoke",
            input_data=input_payload,
            output_data=result,
            duration_ms=node_ms,
        )
        return result

    def __repr__(self) -> str:
        return f"LangGraphAdapter(name={self.name!r})"
