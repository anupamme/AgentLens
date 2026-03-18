# AgentLens Trace Schema

## Design Philosophy

The AgentLens trace schema captures agent behavior at the action level while preserving user privacy. Inspired by Anthropic's Clio system, it records *what agents do* (action types, autonomy levels, outcomes) without storing raw inputs or outputs.

Key principles:

- **Privacy by design** — Inputs are stored only as hashes; outputs are truncated summaries
- **Structured observability** — Every action has a type, autonomy level, and outcome
- **Escalation tracking** — Human-in-the-loop events are first-class citizens
- **Extensibility** — Metadata fields allow domain-specific annotations

## Core Models

### ActionRecord

A single action taken by an agent during a session.

| Field | Type | Description |
|-------|------|-------------|
| `action_id` | `str` | Unique identifier for this action |
| `action_type` | `ActionType` | Category of action (read, write, execute, search, communicate, reason) |
| `autonomy_level` | `AutonomyLevel` | Level of human oversight |
| `outcome` | `ActionOutcome` | Result of the action |
| `timestamp` | `datetime` | When the action occurred (UTC) |
| `duration_ms` | `int` | How long the action took in milliseconds |
| `input_hash` | `str` | Privacy-preserving hash of the input (`xxh64:` or `sha256:` prefix) |
| `output_summary` | `str` | Truncated summary (max 500 chars) |
| `tool_name` | `str?` | Name of external tool used, if any |
| `metadata` | `dict` | Additional key-value pairs |

### EscalationEvent

Records when and why an agent deferred to a human.

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | `datetime` | When the escalation occurred |
| `reason` | `EscalationReason` | Why the agent escalated |
| `action_id` | `str` | Which action triggered escalation |
| `description` | `str` | Brief description (max 200 chars) |
| `resolved` | `bool` | Whether the escalation was resolved |

### SessionTrace

Complete trace of an agent session, containing one or more actions.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `str` | Unique session identifier |
| `agent_id` | `str` | Identifier for the agent |
| `task_category` | `TaskCategory` | High-level task type |
| `session_outcome` | `SessionOutcome` | Overall session result |
| `start_time` | `datetime` | Session start (UTC) |
| `end_time` | `datetime` | Session end (UTC) |
| `actions` | `list[ActionRecord]` | At least one action required |
| `escalations` | `list[EscalationEvent]` | Escalation events (may be empty) |
| `total_tokens` | `int` | Total LLM tokens consumed |
| `schema_version` | `str` | Schema version (currently "1.0") |
| `metadata` | `dict` | Additional session-level data |

## Autonomy Taxonomy

The `AutonomyLevel` enum captures a spectrum of human oversight:

1. **`full_auto`** — Agent acts independently with no human review
2. **`auto_with_audit`** — Agent acts independently but actions are logged for later audit
3. **`human_confirmed`** — Agent proposes an action, human approves before execution
4. **`human_driven`** — Human initiates and directs the action, agent assists

This taxonomy enables analysis of where agents operate autonomously versus where humans maintain control — a key dimension for understanding oversight gaps.

## Connection to Clio

Anthropic's Clio system uses privacy-preserving techniques to understand how Claude is used in practice. AgentLens applies similar principles to *agentic* workflows:

- Like Clio, we hash inputs rather than storing them verbatim
- We add structure that Clio doesn't capture: action types, autonomy levels, escalation events
- Our schema is designed for aggregation — individual traces can be clustered and summarized without exposing private data

## Versioning Policy

The schema uses semantic versioning via the `schema_version` field:

- **Patch** (1.0.x) — Additive metadata fields, no breaking changes
- **Minor** (1.x.0) — New optional fields, new enum values
- **Major** (x.0.0) — Breaking changes to required fields or validation rules

All exported JSON schemas include the version for compatibility checking.
