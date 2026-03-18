# AgentLens

**Privacy-preserving observability framework for LLM agents**, inspired by Anthropic's [Clio](https://www.anthropic.com/research/clio) system.

AgentLens captures structured traces of agent behavior — what actions they take, how autonomous they are, when they escalate to humans — without storing raw user inputs or outputs.

## Quickstart

```bash
pip install -e ".[dev]"
```

### Create a trace

```python
from datetime import datetime, timezone
from agentlens import (
    ActionRecord, SessionTrace, ActionType,
    AutonomyLevel, ActionOutcome, SessionOutcome, TaskCategory,
)
from agentlens.utils.hashing import hash_input

action = ActionRecord(
    action_id="act-001",
    action_type=ActionType.READ,
    autonomy_level=AutonomyLevel.FULL_AUTO,
    outcome=ActionOutcome.SUCCESS,
    timestamp=datetime.now(timezone.utc),
    duration_ms=150,
    input_hash=hash_input("Read the project requirements"),
    output_summary="Read requirements: 12 user stories",
)

session = SessionTrace(
    session_id="sess-001",
    agent_id="my-agent-v1",
    task_category=TaskCategory.CODE_REVIEW,
    session_outcome=SessionOutcome.SUCCESS,
    start_time=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
    end_time=datetime(2025, 1, 1, 12, 5, tzinfo=timezone.utc),
    actions=[action],
)

print(session.to_json())
```

### Export JSON Schema

```bash
python scripts/export_schema.py
```

### Run tests

```bash
pytest tests/ -v
```

## Documentation

- [Schema Design](docs/schema.md) — design philosophy, field reference, autonomy taxonomy
- [Trace Examples](docs/examples.md) — annotated walkthroughs of sample traces

## Project Structure

```
src/agentlens/
    schema/         # Pydantic trace models and enums
    sdk/            # Tracing instrumentation (planned)
    aggregation/    # Aggregation pipeline (planned)
    analysis/       # Analysis module (planned)
    utils/          # Hashing and timestamp utilities
tests/              # Test suite with fixtures
schemas/            # Exported JSON Schema
docs/               # Documentation
examples/           # Example scripts
scripts/            # Utility scripts
```

## Issue Dependency Graph

```
Issue #1: Schema Design (Week 1)
    |
    +-- Issue #2: SDK & Agents (Weeks 2-3)
    |       |
    |       +-- Issue #3: Aggregation Pipeline (Week 4)
    |       |       |
    |       |       +-- Issue #4: Privacy Validation (Weeks 5-6)
    |       |       |
    |       |       +-- Issue #5: Data Collection (Weeks 7-9)
    |       |       |       |
    |       |       |       +-- Issue #6: Analysis & Findings (Weeks 10-11)
    |       |       |               |
    |       |       |               +-- Issue #7: Release & Write-Up (Weeks 12-14)
```

## License

MIT
