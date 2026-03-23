.PHONY: install test lint type-check all mock-pipeline privacy-validation analysis

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src/agentlens --cov-report=term-missing

lint:
	ruff check src/ tests/ scripts/

type-check:
	mypy src/agentlens/

all: lint type-check test

mock-pipeline:
	python -m agentlens run --traces-dir ./traces --output ./reports --mock

privacy-validation:
	python -m agentlens.privacy.runner --traces-dir ./traces --output-dir ./privacy_results --mock

analysis:
	python -m agentlens.analysis run --summaries-dir ./summaries --output ./analysis_results
