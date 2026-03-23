# Contributing to AgentLens

Thank you for your interest in contributing to AgentLens!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/anupamme/AgentLens.git
cd AgentLens

# Install in editable mode with all dev dependencies
pip install -e ".[dev]"

# Optionally install optional extras
pip install -e ".[dev,aggregation,analysis,privacy,workloads]"
```

Verify the install works:

```bash
python -c "import agentlens; print(agentlens.__version__)"
agentlens --help
```

## Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src/agentlens --cov-report=term-missing

# Run a single test file
pytest tests/test_schema.py -v

# Run tests matching a keyword
pytest tests/ -k "pipeline" -v
```

All tests use mock implementations and make no API calls. The full suite should
complete in under 60 seconds.

## Linting and Type Checking

```bash
# Lint with ruff
ruff check src/ tests/ scripts/

# Auto-fix ruff issues
ruff check --fix src/ tests/ scripts/

# Type check with mypy
mypy src/agentlens/
```

Code style is configured in `pyproject.toml`:
- Line length: 100
- Python target: 3.10+
- Ruff rules: E, F, I, N, W, UP (errors, pyflakes, isort, naming, warnings, upgrades)

## Using the Makefile

```bash
make install        # pip install -e ".[dev]"
make test           # pytest with coverage
make lint           # ruff check
make type-check     # mypy
make all            # lint + type-check + test

make mock-pipeline  # run end-to-end pipeline in mock mode
make privacy-validation  # run privacy validation in mock mode
make analysis       # run analysis on existing summaries
```

## Submitting a Pull Request

1. Fork the repository and create a branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes. Add or update tests for any new functionality.

3. Ensure all checks pass:
   ```bash
   make all
   ```

4. Commit using [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat: add support for OpenAI model tracing
   fix: handle empty trace files gracefully
   docs: update quickstart with new CLI flags
   test: add integration test for privacy pipeline
   refactor: extract common summarizer logic
   ```

5. Push and open a PR against `main`. Fill in the PR template describing what
   you changed and why.

## Code Style

- Follow the existing patterns in `src/agentlens/`
- Use `from __future__ import annotations` in all new modules
- Pydantic models go in `models.py`; business logic in separate files
- Abstract base classes end in `Base` (e.g. `BaseSummarizer`)
- Mock implementations are co-located with real ones (e.g. `MockSummarizer`)
- Add type annotations to all public functions

## Adding a New Agent Type

1. Add action sequences to `src/agentlens/workloads/runner.py` (`_ACTION_SEQUENCES`)
2. Add workload templates to `src/agentlens/workloads/mock_generator.py` (`TEMPLATES`)
3. Map the agent type to a `TaskCategory` in `src/agentlens/workloads/generator.py`
4. Add example traces to `tests/fixtures/`

## Project Layout

```
src/agentlens/
    schema/         Core Pydantic models (SessionTrace, ActionRecord, enums)
    sdk/            Instrumentation SDK (AgentTracer, TraceWriter, LangChain)
    aggregation/    Two-stage LLM summarization/aggregation pipeline
    analysis/       Five-dimensional oversight analysis engine
    privacy/        Privacy validation experiments
    workloads/      Synthetic workload generation
    utils/          Hashing and timestamps
```

## Questions?

Open an issue at https://github.com/anupamme/AgentLens/issues
