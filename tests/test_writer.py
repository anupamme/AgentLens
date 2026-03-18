"""Tests for TraceWriter file I/O."""

import os

from agentlens.schema.enums import (
    ActionOutcome,
    ActionType,
    AutonomyLevel,
    SessionOutcome,
    TaskCategory,
)
from agentlens.sdk.tracer import AgentTracer
from agentlens.sdk.writer import TraceWriter


def _make_trace(suffix: str = ""):
    tracer = AgentTracer(
        agent_type=f"test_agent{suffix}",
        task_category=TaskCategory.OTHER,
    )
    tracer.start_session(f"test task{suffix}")
    tracer.record_action(
        action_type=ActionType.READ,
        autonomy_level=AutonomyLevel.FULL_AUTO,
        outcome=ActionOutcome.SUCCESS,
        raw_input=f"input{suffix}",
        output_summary=f"output{suffix}",
        duration_ms=10,
    )
    return tracer.end_session(SessionOutcome.SUCCESS)


class TestJsonlRoundTrip:
    def test_write_and_read_multiple(self, tmp_path):
        writer = TraceWriter(output_dir=str(tmp_path))
        traces = [_make_trace(str(i)) for i in range(5)]

        for t in traces:
            writer.write_jsonl(t)

        loaded = writer.read_traces()
        assert len(loaded) == 5
        for original, loaded_t in zip(traces, loaded):
            assert original.session_id == loaded_t.session_id
            assert original.agent_id == loaded_t.agent_id

    def test_append_behavior(self, tmp_path):
        writer = TraceWriter(output_dir=str(tmp_path))
        t1 = _make_trace("_first")
        t2 = _make_trace("_second")

        writer.write_jsonl(t1)
        writer.write_jsonl(t2)

        loaded = writer.read_traces()
        assert len(loaded) == 2
        assert loaded[0].session_id == t1.session_id
        assert loaded[1].session_id == t2.session_id


class TestJsonSingleFile:
    def test_write_and_read(self, tmp_path):
        writer = TraceWriter(output_dir=str(tmp_path))
        trace = _make_trace()

        path = writer.write_json(trace)
        assert os.path.exists(path)

        loaded = writer.read_trace(f"{trace.session_id}.json")
        assert loaded.session_id == trace.session_id
        assert len(loaded.actions) == len(trace.actions)
        assert loaded.actions[0].input_hash == trace.actions[0].input_hash

    def test_custom_filename(self, tmp_path):
        writer = TraceWriter(output_dir=str(tmp_path))
        trace = _make_trace()

        writer.write_json(trace, filename="custom.json")
        loaded = writer.read_trace("custom.json")
        assert loaded.session_id == trace.session_id
