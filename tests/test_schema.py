"""Tests for the AgentLens trace schema."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentlens.schema.enums import (
    ActionOutcome,
    ActionType,
    AutonomyLevel,
    SessionOutcome,
    TaskCategory,
)
from agentlens.schema.trace import ActionRecord, SessionTrace
from agentlens.utils.hashing import hash_input

FIXTURES = Path(__file__).parent / "fixtures"


def make_action(**overrides):
    defaults = {
        "action_id": "act-001",
        "action_type": ActionType.READ,
        "autonomy_level": AutonomyLevel.FULL_AUTO,
        "outcome": ActionOutcome.SUCCESS,
        "timestamp": datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "duration_ms": 100,
        "input_hash": hash_input("test input"),
        "output_summary": "Test output",
    }
    defaults.update(overrides)
    return ActionRecord(**defaults)


def make_session(**overrides):
    defaults = {
        "session_id": "sess-001",
        "agent_id": "test-agent",
        "task_category": TaskCategory.CODE_REVIEW,
        "session_outcome": SessionOutcome.SUCCESS,
        "start_time": datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "end_time": datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
        "actions": [make_action()],
    }
    defaults.update(overrides)
    return SessionTrace(**defaults)


class TestActionRecord:
    def test_valid_creation(self):
        action = make_action()
        assert action.action_id == "act-001"
        assert action.action_type == ActionType.READ
        assert action.is_autonomous is True

    def test_computed_is_autonomous(self):
        auto = make_action(autonomy_level=AutonomyLevel.FULL_AUTO)
        audit = make_action(autonomy_level=AutonomyLevel.AUTO_WITH_AUDIT)
        confirmed = make_action(autonomy_level=AutonomyLevel.HUMAN_CONFIRMED)
        driven = make_action(autonomy_level=AutonomyLevel.HUMAN_DRIVEN)
        assert auto.is_autonomous is True
        assert audit.is_autonomous is True
        assert confirmed.is_autonomous is False
        assert driven.is_autonomous is False

    def test_input_hash_validation(self):
        with pytest.raises(ValidationError, match="input_hash"):
            make_action(input_hash="invalid_hash")

    def test_input_hash_accepts_sha256(self):
        action = make_action(input_hash=hash_input("data", method="sha256"))
        assert action.input_hash.startswith("sha256:")

    def test_output_truncation(self):
        long_output = "x" * 600
        action = make_action(output_summary=long_output)
        assert len(action.output_summary) == 500
        assert action.output_summary.endswith("...")


class TestSessionTrace:
    def test_valid_creation(self):
        session = make_session()
        assert session.session_id == "sess-001"
        assert len(session.actions) == 1

    def test_computed_properties(self):
        actions = [
            make_action(
                action_id="a1",
                autonomy_level=AutonomyLevel.FULL_AUTO,
                outcome=ActionOutcome.SUCCESS,
            ),
            make_action(
                action_id="a2",
                autonomy_level=AutonomyLevel.HUMAN_CONFIRMED,
                outcome=ActionOutcome.FAILURE,
            ),
            make_action(
                action_id="a3",
                autonomy_level=AutonomyLevel.AUTO_WITH_AUDIT,
                outcome=ActionOutcome.SUCCESS,
            ),
        ]
        session = make_session(actions=actions)
        assert session.duration_ms == 300000  # 5 minutes
        assert abs(session.autonomy_ratio - 2 / 3) < 0.01
        assert abs(session.success_rate - 2 / 3) < 0.01

    def test_action_type_distribution(self):
        actions = [
            make_action(action_id="a1", action_type=ActionType.READ),
            make_action(action_id="a2", action_type=ActionType.READ),
            make_action(action_id="a3", action_type=ActionType.WRITE),
        ]
        session = make_session(actions=actions)
        dist = session.action_type_distribution
        assert dist == {"read": 2, "write": 1}

    def test_minimum_actions_required(self):
        with pytest.raises(ValidationError):
            make_session(actions=[])

    def test_time_order_validation(self):
        with pytest.raises(ValidationError, match="end_time"):
            make_session(
                start_time=datetime(2025, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
                end_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            )

    def test_enum_validation(self):
        with pytest.raises(ValidationError):
            make_session(task_category="invalid_category")

    def test_round_trip_serialization(self):
        session = make_session()
        json_str = session.to_json()
        restored = SessionTrace.from_json(json_str)
        assert restored.session_id == session.session_id
        assert len(restored.actions) == len(session.actions)
        assert restored.actions[0].input_hash == session.actions[0].input_hash

    def test_fixture_valid_trace_01(self):
        data = (FIXTURES / "valid_trace_01.json").read_text()
        session = SessionTrace.from_json(data)
        assert session.session_id == "sess-cr-001"
        assert len(session.actions) == 7
        assert session.session_outcome == SessionOutcome.SUCCESS

    def test_fixture_valid_trace_02(self):
        data = (FIXTURES / "valid_trace_02.json").read_text()
        session = SessionTrace.from_json(data)
        assert session.session_id == "sess-ra-001"
        assert len(session.actions) == 6
        assert len(session.escalations) == 1
        assert session.session_outcome == SessionOutcome.PARTIAL

    def test_fixture_invalid_trace_01(self):
        data = (FIXTURES / "invalid_trace_01.json").read_text()
        with pytest.raises(ValidationError):
            SessionTrace.from_json(data)
