"""
CrewAI Adapter — deep task-level tracing.

Hooks into CrewAI's Crew execution to capture each task as a separate
TraceStep, giving TEI per-task visibility for diagnosis.

Works with CrewAI >=0.60. If CrewAI is not installed, the adapter
still works via generic kickoff wrapping.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Callable, Optional

from ..tracer import TEITracer, _safe_serialize
from ..models import Trace, TraceStep


class CrewAIAdapter:
    """Wraps a CrewAI Crew for TEI with per-task tracing."""

    def __init__(self, crew: Any, name: str = "crewai_agent"):
        self.crew = crew
        self.name = name
        self._tracer = TEITracer()

    async def run(self, query: Any, context: Optional[dict[str, Any]] = None) -> Trace:
        self._tracer.start()
        overall_start = time.time()

        inputs = {"query": str(query)}
        if context:
            inputs.update(context)

        output = None
        error_msg = None

        try:
            self._trace_crew_config()

            if self._has_task_callbacks():
                output = await self._run_with_callbacks(inputs)
            elif hasattr(self.crew, "kickoff_async"):
                output = await self._run_async(inputs)
            else:
                output = await self._run_sync(inputs)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"

        overall_ms = (time.time() - overall_start) * 1000
        self._tracer.add_step(
            name="crewai_total",
            step_type="orchestrator",
            input_data=query,
            output_data=output,
            duration_ms=overall_ms,
            error=error_msg,
            metadata={
                "adapter": "crewai",
                "task_count": self._get_task_count(),
                "agent_count": self._get_agent_count(),
            },
        )

        return self._tracer.finish(agent_input=query, agent_output=output)

    def _trace_crew_config(self) -> None:
        """Capture crew configuration as a trace step for diagnostic context."""
        agents_info = []
        if hasattr(self.crew, "agents"):
            for ag in self.crew.agents:
                agents_info.append({
                    "role": getattr(ag, "role", "unknown"),
                    "goal": str(getattr(ag, "goal", ""))[:200],
                    "backstory": str(getattr(ag, "backstory", ""))[:200],
                    "tools": [
                        getattr(t, "name", str(t))
                        for t in getattr(ag, "tools", [])
                    ],
                })

        tasks_info = []
        if hasattr(self.crew, "tasks"):
            for task in self.crew.tasks:
                tasks_info.append({
                    "description": str(getattr(task, "description", ""))[:200],
                    "expected_output": str(getattr(task, "expected_output", ""))[:200],
                    "agent_role": getattr(
                        getattr(task, "agent", None), "role", "unassigned"
                    ),
                })

        self._tracer.add_step(
            name="crew_config",
            step_type="crewai_config",
            metadata={
                "agents": agents_info,
                "tasks": tasks_info,
                "process": str(getattr(self.crew, "process", "sequential")),
            },
        )

    def _has_task_callbacks(self) -> bool:
        """Check if crew tasks support callback injection."""
        if not hasattr(self.crew, "tasks"):
            return False
        tasks = self.crew.tasks
        if not tasks:
            return False
        return hasattr(tasks[0], "callback")

    async def _run_with_callbacks(self, inputs: dict[str, Any]) -> Any:
        """Inject per-task callbacks to capture task-level output."""
        task_results: list[dict[str, Any]] = []

        original_callbacks = []
        for i, task in enumerate(self.crew.tasks):
            original_callbacks.append(getattr(task, "callback", None))

            task_name = getattr(
                getattr(task, "agent", None), "role", f"task_{i}"
            )
            task_desc = str(getattr(task, "description", ""))[:200]

            def make_callback(idx: int, name: str, desc: str):
                task_start = time.time()

                def cb(output: Any) -> None:
                    task_ms = (time.time() - task_start) * 1000
                    raw_output = getattr(output, "raw", str(output))

                    self._tracer.add_step(
                        name=name,
                        step_type="crewai_task",
                        input_data=desc,
                        output_data=raw_output,
                        duration_ms=task_ms,
                        metadata={
                            "task_index": idx,
                            "agent_role": name,
                        },
                    )
                    task_results.append({
                        "task_index": idx,
                        "agent": name,
                        "output_preview": str(raw_output)[:300],
                    })

                return cb

            task.callback = make_callback(i, task_name, task_desc)

        try:
            if hasattr(self.crew, "kickoff_async"):
                result = await self.crew.kickoff_async(inputs=inputs)
            else:
                result = await asyncio.to_thread(
                    self.crew.kickoff, inputs=inputs
                )
        finally:
            for i, task in enumerate(self.crew.tasks):
                task.callback = original_callbacks[i]

        return self._extract_output(result)

    async def _run_async(self, inputs: dict[str, Any]) -> Any:
        """Fallback: async kickoff without per-task callbacks."""
        task_start = time.time()
        result = await self.crew.kickoff_async(inputs=inputs)
        task_ms = (time.time() - task_start) * 1000

        self._tracer.add_step(
            name="kickoff_async",
            step_type="crewai_kickoff",
            input_data=inputs,
            output_data=self._extract_output(result),
            duration_ms=task_ms,
        )

        self._trace_task_outputs(result)
        return self._extract_output(result)

    async def _run_sync(self, inputs: dict[str, Any]) -> Any:
        """Fallback: sync kickoff wrapped in thread."""
        task_start = time.time()
        result = await asyncio.to_thread(self.crew.kickoff, inputs=inputs)
        task_ms = (time.time() - task_start) * 1000

        self._tracer.add_step(
            name="kickoff",
            step_type="crewai_kickoff",
            input_data=inputs,
            output_data=self._extract_output(result),
            duration_ms=task_ms,
        )

        self._trace_task_outputs(result)
        return self._extract_output(result)

    def _trace_task_outputs(self, crew_output: Any) -> None:
        """Extract per-task outputs from CrewOutput (post-hoc tracing)."""
        tasks_output = getattr(crew_output, "tasks_output", None)
        if not tasks_output:
            return

        for i, task_out in enumerate(tasks_output):
            agent_name = getattr(task_out, "agent", f"task_{i}")
            raw = getattr(task_out, "raw", str(task_out))

            self._tracer.add_step(
                name=str(agent_name),
                step_type="crewai_task_output",
                output_data=raw,
                metadata={
                    "task_index": i,
                    "description": str(
                        getattr(task_out, "description", "")
                    )[:200],
                },
            )

    def _extract_output(self, result: Any) -> Any:
        if hasattr(result, "raw"):
            return result.raw
        return str(result)

    def _get_task_count(self) -> int:
        return len(getattr(self.crew, "tasks", []))

    def _get_agent_count(self) -> int:
        return len(getattr(self.crew, "agents", []))

    def __repr__(self) -> str:
        return f"CrewAIAdapter(name={self.name!r})"
