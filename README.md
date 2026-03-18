# AgentLens — GitHub Issues Index

## Project Overview

**AgentLens** is a privacy-preserving observability framework for LLM agents, inspired by Anthropic's Clio system. 

## Issue Dependency Graph

```
Issue #1: Schema Design (Week 1)
    │
    ├── Issue #2: SDK & Agents (Weeks 2-3)
    │       │
    │       ├── Issue #3: Aggregation Pipeline (Week 4)
    │       │       │
    │       │       ├── Issue #4: Privacy Validation (Weeks 5-6)
    │       │       │
    │       │       ├── Issue #5: Data Collection (Weeks 7-9)
    │       │       │       │
    │       │       │       └── Issue #6: Analysis & Findings (Weeks 10-11)
    │       │       │               │
    │       │       │               └── Issue #7: Release & Write-Up (Weeks 12-14)
    │       │       │
    │       │       └───────────────────┘
    │       │
    │       └───────────────────────────┘
    │
    └───────────────────────────────────┘
```

## Issues Summary

| # | Title | Phase | Weeks | Key Deliverable |
|---|-------|-------|-------|-----------------|
| 1 | [Schema Design](./issue-01-schema-design.md) | Phase 1 | 1 | Pydantic trace schema, JSON Schema export, test fixtures |
| 2 | [SDK & Agents](./issue-02-sdk-agents.md) | Phase 1 | 2–3 | AgentTracer SDK, LangChain integration, 3 working agents |
| 3 | [Aggregation Pipeline](./issue-03-aggregation-pipeline.md) | Phase 2 | 4 | Two-stage LLM pipeline, CLI, mock mode |
| 4 | [Privacy Validation](./issue-04-privacy-validation.md) | Phase 2 | 5–6 | PII leakage test, re-identification attack, utility-privacy curves |
| 5 | [Data Collection](./issue-05-data-collection.md) | Phase 3 | 7–9 | Workload generator, 500–2,000 validated traces |
| 6 | [Analysis & Findings](./issue-06-analysis.md) | Phase 4 | 10–11 | Five-dimensional analysis, 10+ plots, oversight gap score |
| 7 | [Release & Write-Up](./issue-07-release-writeup.md) | Phase 5 | 12–14 | Polished repo, HuggingFace dataset, research write-up |
