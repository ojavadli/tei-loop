"""Tests for TEI tracer module."""

import asyncio
import pytest
from tei_loop.tracer import TEITracer, run_and_trace, tei_trace, _safe_serialize


def test_tracer_basic():
    tracer = TEITracer()
    tracer.start()
    tracer.add_step(name="step1", input_data="hello", output_data="world", duration_ms=100)
    trace = tracer.finish(agent_input="hello", agent_output="world")
    assert len(trace.steps) == 1
    assert trace.steps[0].name == "step1"
    assert trace.agent_input == "hello"


@pytest.mark.asyncio
async def test_run_and_trace_sync():
    def my_agent(query):
        return f"Answer to: {query}"

    trace = await run_and_trace(my_agent, "What is 2+2?")
    assert trace.agent_input == "What is 2+2?"
    assert "Answer to:" in str(trace.agent_output)
    assert len(trace.steps) >= 1


@pytest.mark.asyncio
async def test_run_and_trace_async():
    async def my_agent(query):
        return f"Async answer to: {query}"

    trace = await run_and_trace(my_agent, "Test query")
    assert "Async answer" in str(trace.agent_output)


@pytest.mark.asyncio
async def test_run_and_trace_error():
    def failing_agent(query):
        raise ValueError("Agent crashed")

    trace = await run_and_trace(failing_agent, "test")
    assert trace.agent_output is None
    assert trace.steps[0].error
    assert "ValueError" in trace.steps[0].error


def test_safe_serialize_truncation():
    long_string = "x" * 10000
    result = _safe_serialize(long_string, max_len=100)
    assert len(result) < 200
    assert "truncated" in result


def test_safe_serialize_dict():
    data = {"key": "value", "nested": {"inner": 42}}
    result = _safe_serialize(data)
    assert result["key"] == "value"
    assert result["nested"]["inner"] == 42
