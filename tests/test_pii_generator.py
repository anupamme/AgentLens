"""Tests for PII generation and trace injection."""

from dataclasses import fields
from datetime import datetime, timezone

import pytest

from agentlens.privacy.pii_generator import PIIGenerator, SyntheticPII
from agentlens.schema.enums import (
    ActionOutcome,
    ActionType,
    AutonomyLevel,
    EscalationReason,
    SessionOutcome,
    TaskCategory,
)
from agentlens.schema.trace import ActionRecord, EscalationEvent, SessionTrace
from agentlens.utils.hashing import hash_input


def _make_action(**overrides):
    defaults = {
        "action_id": "act-001",
        "action_type": ActionType.READ,
        "autonomy_level": AutonomyLevel.FULL_AUTO,
        "outcome": ActionOutcome.SUCCESS,
        "timestamp": datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "duration_ms": 100,
        "input_hash": hash_input("test input"),
        "output_summary": "Test output summary",
    }
    defaults.update(overrides)
    return ActionRecord(**defaults)


def _make_session(**overrides):
    defaults = {
        "session_id": "sess-001",
        "agent_id": "test-agent",
        "task_category": TaskCategory.CODE_REVIEW,
        "session_outcome": SessionOutcome.SUCCESS,
        "start_time": datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "end_time": datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
        "actions": [_make_action()],
    }
    defaults.update(overrides)
    return SessionTrace(**defaults)


class TestPIIGenerator:
    def test_generate_50_unique_bundles(self):
        gen = PIIGenerator(seed=42)
        bundles = gen.generate(count=50)

        assert len(bundles) == 50

        # All unique emails
        emails = [b.email for b in bundles]
        assert len(set(emails)) == 50

        # All fields non-empty
        for bundle in bundles:
            for f in fields(bundle):
                val = getattr(bundle, f.name)
                assert val, f"Field {f.name} is empty"

    def test_deterministic_with_same_seed(self):
        gen1 = PIIGenerator(seed=123)
        gen2 = PIIGenerator(seed=123)

        bundles1 = gen1.generate(count=10)
        bundles2 = gen2.generate(count=10)

        for b1, b2 in zip(bundles1, bundles2):
            assert b1.full_name == b2.full_name
            assert b1.email == b2.email
            assert b1.api_key == b2.api_key

    def test_different_seeds_produce_different_results(self):
        gen1 = PIIGenerator(seed=1)
        gen2 = PIIGenerator(seed=2)

        bundles1 = gen1.generate(count=5)
        bundles2 = gen2.generate(count=5)

        # At least some should differ
        emails1 = {b.email for b in bundles1}
        emails2 = {b.email for b in bundles2}
        assert emails1 != emails2

    def test_inject_into_trace_original_unchanged(self):
        gen = PIIGenerator(seed=42)
        pii = gen.generate(count=1)[0]

        trace = _make_session()
        original_output = trace.actions[0].output_summary
        original_metadata = dict(trace.metadata)

        injected = gen.inject_into_trace(trace, pii)

        # Original unchanged
        assert trace.actions[0].output_summary == original_output
        assert trace.metadata == original_metadata

        # Injected has PII
        assert pii.full_name in injected.actions[0].output_summary
        assert pii.email in injected.actions[0].output_summary
        assert injected.metadata["user_context"] == pii.home_address

    def test_inject_respects_output_summary_limit(self):
        gen = PIIGenerator(seed=42)
        pii = gen.generate(count=1)[0]

        # Create trace with long output_summary
        trace = _make_session(actions=[
            _make_action(output_summary="x" * 400),
        ])

        injected = gen.inject_into_trace(trace, pii)
        assert len(injected.actions[0].output_summary) <= 500

    def test_inject_respects_escalation_description_limit(self):
        gen = PIIGenerator(seed=42)
        pii = gen.generate(count=1)[0]

        trace = _make_session(
            actions=[_make_action(action_id="a1", outcome=ActionOutcome.FAILURE)],
            escalations=[
                EscalationEvent(
                    timestamp=datetime(2025, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
                    reason=EscalationReason.ERROR_REPEATED,
                    action_id="a1",
                    description="A" * 180,
                ),
            ],
        )

        injected = gen.inject_into_trace(trace, pii)
        assert len(injected.escalations[0].description) <= 200

    def test_get_all_pii_strings_includes_partials(self):
        pii = SyntheticPII(
            full_name="John Smith",
            email="john.smith42@gmail.com",
            phone="+1-555-123-4567",
            ssn="123-45-6789",
            api_key="sk-abcdefghijklmnopqrstuvwxyz1234567890ABCD",
            credit_card="1234-5678-9012-3456",
            ip_address="192.168.1.100",
            github_username="johnsmith42",
            home_address="123 Oak Street, Springfield, IL 62704",
            code_snippet='db_password = "secret123"',
            file_path="/home/john/projects/myapp/config.env",
            aws_secret="AKIAIOSFODNN7EXAMPLE+secret",
        )

        strings = PIIGenerator.get_all_pii_strings(pii)

        # Full values
        assert pii.full_name in strings
        assert pii.email in strings
        assert pii.phone in strings

        # Partial: first name, last name
        assert "John" in strings
        assert "Smith" in strings

        # Partial: email local part
        assert "john.smith42" in strings

        # Partial: first 11 chars of API key (sk- + 8)
        assert "sk-abcdefgh" in strings

    def test_diverse_name_regions(self):
        gen = PIIGenerator(seed=42)
        bundles = gen.generate(count=12)  # 2 per region

        names = [b.full_name for b in bundles]
        # Should have names from multiple regions (at least some variety)
        assert len(set(names)) > 6
