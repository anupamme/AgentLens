"""Mock workload generator — template-based, no LLM calls."""

from __future__ import annotations

import random
from typing import Any
from uuid import uuid4

from agentlens.workloads.generator import Difficulty, FailureMode, TaskConfig

_FAILURE_MODES_WITH_INJECTION = [
    FailureMode.TOOL_TIMEOUT,
    FailureMode.AMBIGUOUS_INPUT,
    FailureMode.CONFLICTING_CONSTRAINTS,
    FailureMode.SAFETY_BOUNDARY,
    FailureMode.PARTIAL_FAILURE,
]

TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "code_reviewer": [
        {"prompt": "Review the authentication module for SQL injection vulnerabilities and suggest fixes.", "difficulty": "hard", "expected_tool_count": 5, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "security"}},
        {"prompt": "Check the REST API endpoints for proper error handling and response codes.", "difficulty": "medium", "expected_tool_count": 4, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "api"}},
        {"prompt": "Review the database migration scripts for backward compatibility issues.", "difficulty": "hard", "expected_tool_count": 3, "expected_autonomy_pattern": "human_guided", "metadata": {"domain": "database"}},
        {"prompt": "Examine the logging configuration for sensitive data exposure.", "difficulty": "easy", "expected_tool_count": 2, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "security"}},
        {"prompt": "Check the unit tests for adequate coverage of edge cases in the payment module.", "difficulty": "medium", "expected_tool_count": 4, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "testing"}},
        {"prompt": "Review the CSS stylesheets for unused selectors and performance issues.", "difficulty": "easy", "expected_tool_count": 2, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "frontend"}},
        {"prompt": "Evaluate the error handling strategy in the file upload service.", "difficulty": "medium", "expected_tool_count": 3, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "backend"}},
        {"prompt": "Review the caching implementation for potential race conditions.", "difficulty": "hard", "expected_tool_count": 5, "expected_autonomy_pattern": "human_guided", "metadata": {"domain": "performance"}},
        {"prompt": "Check the configuration management code for hardcoded secrets.", "difficulty": "easy", "expected_tool_count": 2, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "security"}},
        {"prompt": "Review the data serialization layer for type safety issues.", "difficulty": "medium", "expected_tool_count": 3, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "data"}},
        {"prompt": "Evaluate the retry logic in the external API client for correctness.", "difficulty": "medium", "expected_tool_count": 3, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "resilience"}},
        {"prompt": "Review the WebSocket implementation for proper connection lifecycle management.", "difficulty": "hard", "expected_tool_count": 4, "expected_autonomy_pattern": "human_guided", "metadata": {"domain": "networking"}},
        {"prompt": "Check the input validation middleware for bypass vulnerabilities.", "difficulty": "hard", "expected_tool_count": 5, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "security"}},
        {"prompt": "Review the batch processing pipeline for memory leak potential.", "difficulty": "medium", "expected_tool_count": 3, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "performance"}},
        {"prompt": "Evaluate the notification service for proper message queuing and delivery.", "difficulty": "medium", "expected_tool_count": 4, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "messaging"}},
        {"prompt": "Check the frontend form validation for XSS vulnerability patterns.", "difficulty": "easy", "expected_tool_count": 2, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "security"}},
        {"prompt": "Review the deployment scripts for idempotency and rollback support.", "difficulty": "hard", "expected_tool_count": 4, "expected_autonomy_pattern": "human_guided", "metadata": {"domain": "devops"}},
        {"prompt": "Check the pagination implementation for off-by-one errors.", "difficulty": "easy", "expected_tool_count": 2, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "api"}},
        {"prompt": "Review the authentication token refresh logic for timing vulnerabilities.", "difficulty": "hard", "expected_tool_count": 4, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "security"}},
        {"prompt": "Evaluate the search indexing code for consistency with the data model.", "difficulty": "medium", "expected_tool_count": 3, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "search"}},
    ],
    "research_assistant": [
        {"prompt": "Compare the top three Python web frameworks for building REST APIs and summarize trade-offs.", "difficulty": "medium", "expected_tool_count": 4, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "technology"}},
        {"prompt": "Research current best practices for securing containerized microservices.", "difficulty": "hard", "expected_tool_count": 5, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "security"}},
        {"prompt": "Find and summarize recent papers on transformer architecture optimizations.", "difficulty": "hard", "expected_tool_count": 6, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "ml"}},
        {"prompt": "Compile a summary of GDPR compliance requirements for SaaS applications.", "difficulty": "medium", "expected_tool_count": 3, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "legal"}},
        {"prompt": "Research the differences between event-driven and request-driven architectures.", "difficulty": "medium", "expected_tool_count": 4, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "architecture"}},
        {"prompt": "Summarize the state of quantum computing applications in cryptography.", "difficulty": "hard", "expected_tool_count": 5, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "science"}},
        {"prompt": "Find the latest benchmarks for vector database performance.", "difficulty": "easy", "expected_tool_count": 3, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "database"}},
        {"prompt": "Research accessibility standards for web applications and create a checklist.", "difficulty": "medium", "expected_tool_count": 4, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "frontend"}},
        {"prompt": "Compare serverless deployment options across major cloud providers.", "difficulty": "medium", "expected_tool_count": 4, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "cloud"}},
        {"prompt": "Research the impact of technical debt on software project velocity.", "difficulty": "easy", "expected_tool_count": 3, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "engineering"}},
        {"prompt": "Summarize the pros and cons of monorepo vs polyrepo strategies.", "difficulty": "easy", "expected_tool_count": 3, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "devops"}},
        {"prompt": "Research the current state of WebAssembly for server-side applications.", "difficulty": "medium", "expected_tool_count": 4, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "technology"}},
        {"prompt": "Find and compare open-source observability tools for distributed systems.", "difficulty": "medium", "expected_tool_count": 5, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "infrastructure"}},
        {"prompt": "Research emerging patterns in API design: GraphQL, gRPC, and REST evolution.", "difficulty": "hard", "expected_tool_count": 5, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "api"}},
        {"prompt": "Compile a summary of machine learning model serving best practices.", "difficulty": "medium", "expected_tool_count": 4, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "ml"}},
        {"prompt": "Research the environmental impact of large language model training.", "difficulty": "medium", "expected_tool_count": 4, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "ethics"}},
        {"prompt": "Find industry standards for API rate limiting and throttling.", "difficulty": "easy", "expected_tool_count": 2, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "api"}},
        {"prompt": "Research zero-trust security architecture implementation strategies.", "difficulty": "hard", "expected_tool_count": 5, "expected_autonomy_pattern": "human_guided", "metadata": {"domain": "security"}},
        {"prompt": "Compare different approaches to database sharding for horizontal scaling.", "difficulty": "hard", "expected_tool_count": 4, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "database"}},
        {"prompt": "Summarize recent developments in federated learning and privacy-preserving ML.", "difficulty": "hard", "expected_tool_count": 5, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "ml"}},
    ],
    "task_manager": [
        {"prompt": "Create a sprint plan for the next two weeks from the current backlog of 15 items.", "difficulty": "medium", "expected_tool_count": 4, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "project_management"}},
        {"prompt": "Generate a status report for stakeholders on the Q3 platform migration.", "difficulty": "easy", "expected_tool_count": 3, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "reporting"}},
        {"prompt": "Identify and resolve dependency conflicts between three concurrent feature branches.", "difficulty": "hard", "expected_tool_count": 5, "expected_autonomy_pattern": "human_guided", "metadata": {"domain": "coordination"}},
        {"prompt": "Prioritize the bug backlog based on severity, customer impact, and effort estimates.", "difficulty": "medium", "expected_tool_count": 3, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "triage"}},
        {"prompt": "Schedule code review assignments for the team for the coming week.", "difficulty": "easy", "expected_tool_count": 2, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "scheduling"}},
        {"prompt": "Create a release checklist for the upcoming v2.0 deployment.", "difficulty": "medium", "expected_tool_count": 4, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "release"}},
        {"prompt": "Analyze team velocity trends and suggest process improvements.", "difficulty": "hard", "expected_tool_count": 5, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "analytics"}},
        {"prompt": "Set up milestone tracking for a six-month product roadmap.", "difficulty": "medium", "expected_tool_count": 4, "expected_autonomy_pattern": "human_guided", "metadata": {"domain": "planning"}},
        {"prompt": "Triage incoming support tickets and assign to appropriate team members.", "difficulty": "easy", "expected_tool_count": 3, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "triage"}},
        {"prompt": "Create a risk assessment matrix for the infrastructure migration project.", "difficulty": "hard", "expected_tool_count": 5, "expected_autonomy_pattern": "human_guided", "metadata": {"domain": "risk"}},
        {"prompt": "Generate a capacity planning report for the engineering team.", "difficulty": "medium", "expected_tool_count": 4, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "planning"}},
        {"prompt": "Coordinate cross-team dependencies for the API versioning initiative.", "difficulty": "hard", "expected_tool_count": 5, "expected_autonomy_pattern": "human_guided", "metadata": {"domain": "coordination"}},
        {"prompt": "Update project timelines based on the latest sprint retrospective outcomes.", "difficulty": "easy", "expected_tool_count": 3, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "planning"}},
        {"prompt": "Create an onboarding task list for a new team member joining the backend team.", "difficulty": "easy", "expected_tool_count": 2, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "onboarding"}},
        {"prompt": "Analyze blocked tasks and propose unblocking strategies for the current sprint.", "difficulty": "medium", "expected_tool_count": 4, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "triage"}},
        {"prompt": "Prepare a technical debt assessment and prioritization for the next quarter.", "difficulty": "hard", "expected_tool_count": 5, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "planning"}},
        {"prompt": "Track and report on SLA compliance for the customer-facing services.", "difficulty": "medium", "expected_tool_count": 3, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "reporting"}},
        {"prompt": "Create a resource allocation plan for three parallel feature development streams.", "difficulty": "hard", "expected_tool_count": 4, "expected_autonomy_pattern": "human_guided", "metadata": {"domain": "planning"}},
        {"prompt": "Review and update the team's definition of done for user stories.", "difficulty": "easy", "expected_tool_count": 2, "expected_autonomy_pattern": "mixed", "metadata": {"domain": "process"}},
        {"prompt": "Generate a weekly digest of completed, in-progress, and blocked items.", "difficulty": "easy", "expected_tool_count": 3, "expected_autonomy_pattern": "fully_autonomous", "metadata": {"domain": "reporting"}},
    ],
}


class MockWorkloadGenerator:
    """Template-based workload generator. No LLM calls."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def generate(
        self,
        agent_type: str,
        count: int = 50,
        failure_injection_rate: float = 0.2,
    ) -> list[TaskConfig]:
        """Generate task configs by cycling through templates with variation."""
        templates = TEMPLATES.get(agent_type, TEMPLATES["task_manager"])
        tasks: list[TaskConfig] = []

        for i in range(count):
            template = templates[i % len(templates)]
            task = TaskConfig(
                task_id=f"{agent_type}_{i:04d}_{uuid4().hex[:8]}",
                agent_type=agent_type,
                prompt=template["prompt"],
                difficulty=Difficulty(template["difficulty"]),
                expected_autonomy_pattern=template["expected_autonomy_pattern"],
                expected_tool_count=template["expected_tool_count"],
                metadata=dict(template.get("metadata", {})),
            )

            # Apply failure injection
            if self._rng.random() < failure_injection_rate:
                task.injected_failure_mode = self._rng.choice(_FAILURE_MODES_WITH_INJECTION)

            tasks.append(task)

        return tasks
