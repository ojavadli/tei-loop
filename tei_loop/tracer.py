"""
TEI Tracer Module.

Captures agent execution as a structured Trace. Three levels of instrumentation:

1. Black-box (default): wraps agent as callable, captures input/output only.
   Works with ANY agent, zero code changes needed.
2. Decorator-based: user adds @tei_trace to key functions for step-level visibility.
3. Framework adapters: auto-hooks into LangGraph/CrewAI nodes for deep tracing.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from typing import Any, Callable, Optional

from .models import Trace, TraceStep


class TEITracer:
    """Captures agent execution into a structured Trace."""

    def __init__(self):
        self._steps: list[TraceStep] = []
        self._start_time: float = 0.0

    def start(self) -> None:
        self._steps = []
        self._start_time = time.time()

    def add_step(
        self,
        name: str,
        step_type: str = "generic",
        input_data: Any = None,
        output_data: Any = None,
        duration_ms: float = 0.0,
        error: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TraceStep:
        step = TraceStep(
            name=name,
            step_type=step_type,
            input_data=_safe_serialize(input_data),
            output_data=_safe_serialize(output_data),
            duration_ms=duration_ms,
            error=error,
            metadata=metadata or {},
        )
        self._steps.append(step)
        return step

    def finish(self, agent_input: Any, agent_output: Any) -> Trace:
        total_ms = (time.time() - self._start_time) * 1000 if self._start_time else 0.0
        return Trace(
            agent_input=_safe_serialize(agent_input),
            agent_output=_safe_serialize(agent_output),
            steps=list(self._steps),
            total_duration_ms=total_ms,
        )


def _run_with_tracer(
    tracer: "TEITracer",
    agent_fn: Callable,
    query: Any,
    ctx: dict[str, Any],
) -> Any:
    """Run a sync agent in a worker thread with the tracer propagated."""
    set_active_tracer(tracer)
    try:
        return agent_fn(query, **ctx) if ctx else agent_fn(query)
    finally:
        clear_active_tracer()


async def run_and_trace(
    agent_fn: Callable,
    query: Any,
    tracer: Optional[TEITracer] = None,
    context: Optional[dict[str, Any]] = None,
) -> Trace:
    """Run an agent function and capture a Trace.

    Handles both sync and async callables. This is the primary black-box
    tracing mechanism: no code changes needed in the agent.
    """
    if tracer is None:
        tracer = TEITracer()

    tracer.start()
    set_active_tracer(tracer)
    ctx = context or {}

    start = time.time()
    error_msg = None
    output = None

    try:
        if inspect.iscoroutinefunction(agent_fn):
            output = await agent_fn(query, **ctx) if ctx else await agent_fn(query)
        else:
            output = await asyncio.to_thread(
                _run_with_tracer, tracer, agent_fn, query, ctx
            )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        output = None
    finally:
        clear_active_tracer()

    duration_ms = (time.time() - start) * 1000
    tracer.add_step(
        name="agent_run",
        step_type="agent_call",
        input_data=query,
        output_data=output,
        duration_ms=duration_ms,
        error=error_msg,
    )

    return tracer.finish(agent_input=query, agent_output=output)


def tei_trace(name: Optional[str] = None, step_type: str = "generic"):
    """Decorator for opt-in step-level tracing inside an agent.

    Usage:
        @tei_trace("search_restaurants", step_type="tool_call")
        def search_restaurants(query):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        fn_name = name or fn.__name__

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            tracer = _get_active_tracer()
            start = time.time()
            error_msg = None
            result = None
            try:
                result = await fn(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                raise
            finally:
                if tracer:
                    tracer.add_step(
                        name=fn_name,
                        step_type=step_type,
                        input_data=_safe_serialize(args[0] if args else kwargs),
                        output_data=_safe_serialize(result),
                        duration_ms=(time.time() - start) * 1000,
                        error=error_msg,
                    )
            return result

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            tracer = _get_active_tracer()
            start = time.time()
            error_msg = None
            result = None
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                raise
            finally:
                if tracer:
                    tracer.add_step(
                        name=fn_name,
                        step_type=step_type,
                        input_data=_safe_serialize(args[0] if args else kwargs),
                        output_data=_safe_serialize(result),
                        duration_ms=(time.time() - start) * 1000,
                        error=error_msg,
                    )
            return result

        if inspect.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator


# ---------------------------------------------------------------------------
# Thread-local active tracer (for decorator-based tracing)
# ---------------------------------------------------------------------------

import threading

_local = threading.local()


def set_active_tracer(tracer: TEITracer) -> None:
    _local.tracer = tracer


def _get_active_tracer() -> Optional[TEITracer]:
    return getattr(_local, "tracer", None)


def clear_active_tracer() -> None:
    _local.tracer = None


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------

def _safe_serialize(obj: Any, max_len: int = 5000) -> Any:
    """Convert an object to a JSON-safe representation, truncating if needed."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + f"... [truncated, {len(obj)} chars total]"
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v, max_len) for k, v in list(obj.items())[:50]}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v, max_len) for v in obj[:50]]
    try:
        s = str(obj)
        if len(s) > max_len:
            return s[:max_len] + f"... [truncated, {len(s)} chars total]"
        return s
    except Exception:
        return f"<{type(obj).__name__}>"
