"""Research Assistant Agent — plans queries, searches, synthesizes findings.

Uses AnthropicBedrock client with AgentLens tracing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config import ANTHROPIC_MODEL_ID, AWS_REGION, MAX_TOKENS_PER_CALL, TRACES_DIR

from agentlens.schema.enums import (
    ActionOutcome,
    ActionType,
    AutonomyLevel,
    EscalationReason,
    SessionOutcome,
    TaskCategory,
)
from agentlens.sdk.tracer import AgentTracer

CANNED_SEARCH_RESULTS = {
    "LLM agent safety": [
        {
            "title": "Constitutional AI: Harmlessness from AI Feedback",
            "snippet": "RLHF + CAI reduces harmful outputs by 40%.",
        },
        {
            "title": "Risks of autonomous LLM agents",
            "snippet": "Uncontrolled tool use can lead to data exfiltration.",
        },
    ],
    "LLM agent oversight mechanisms": [
        {
            "title": "Human-in-the-loop for LLM agents",
            "snippet": "HITL reduces error rates by 60% in high-risk tasks.",
        },
        {
            "title": "Automated guardrails for agents",
            "snippet": "Rule-based guardrails catch 80% of policy violations.",
        },
    ],
    "LLM agent failure modes": [
        {
            "title": "When LLM agents go wrong",
            "snippet": "Agents may hallucinate tool calls, contradicting safety.",
        },
    ],
}

RESEARCH_TOPIC = "Safety and oversight mechanisms for autonomous LLM agents"


def run() -> None:
    tracer = AgentTracer(
        agent_type="research_assistant",
        task_category=TaskCategory.RESEARCH,
        model_used=ANTHROPIC_MODEL_ID,
    )
    tracer.start_session(f"Research: {RESEARCH_TOPIC}")

    # Step 1: Plan search queries with Claude
    queries = list(CANNED_SEARCH_RESULTS.keys())
    try:
        from anthropic import AnthropicBedrock

        client = AnthropicBedrock(aws_region=AWS_REGION)
        with tracer.action(
            action_type=ActionType.REASON,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            raw_input=f"Plan search queries for: {RESEARCH_TOPIC}",
            tool_name="claude_plan",
        ) as ctx:
            response = client.messages.create(
                model=ANTHROPIC_MODEL_ID,
                max_tokens=MAX_TOKENS_PER_CALL,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Generate 3 search queries to research: "
                            f"{RESEARCH_TOPIC}. Return one per line."
                        ),
                    }
                ],
            )
            planned = response.content[0].text.strip().split("\n")[:3]
            if planned:
                queries = planned
            ctx.set_output_summary(f"Planned {len(queries)} queries")
    except Exception:
        tracer.record_action(
            action_type=ActionType.REASON,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            outcome=ActionOutcome.SUCCESS,
            raw_input=f"Plan search queries for: {RESEARCH_TOPIC}",
            output_summary=f"Planned {len(queries)} queries (simulated)",
            duration_ms=30,
            tool_name="claude_plan_simulated",
        )

    # Step 2: Execute searches
    all_results = []
    for query in queries:
        with tracer.action(
            action_type=ActionType.SEARCH,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            raw_input=query,
            tool_name="web_search",
        ) as ctx:
            results = CANNED_SEARCH_RESULTS.get(query, [{"title": "No results", "snippet": "N/A"}])
            all_results.extend(results)
            ctx.set_output_summary(f"Found {len(results)} results for '{query[:50]}'")

    # Step 3: Summarize each source
    summaries = []
    for result in all_results:
        with tracer.action(
            action_type=ActionType.REASON,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            raw_input=f"Summarize: {result['title']} - {result['snippet']}",
        ) as ctx:
            summary = f"{result['title']}: {result['snippet']}"
            summaries.append(summary)
            ctx.set_output_summary(summary[:200])

    # Step 4: Detect contradiction → escalate
    tracer.record_escalation(
        reason=EscalationReason.CONFIDENCE_LOW,
        context_summary=(
            "Sources contradict on whether automated guardrails "
            "are sufficient for safety"
        ),
    )

    # Step 5: Synthesize
    with tracer.action(
        action_type=ActionType.REASON,
        autonomy_level=AutonomyLevel.FULL_AUTO,
        raw_input="Synthesize all summaries: " + "; ".join(summaries),
    ) as ctx:
        ctx.set_output_summary(
            "LLM agent safety requires layered approach: RLHF/CAI for alignment, "
            "HITL for high-risk decisions, guardrails for policy compliance. "
            "Sources disagree on guardrail sufficiency."
        )

    trace = tracer.end_session(SessionOutcome.PARTIAL, user_satisfaction_proxy=0.7)

    output_dir = str(TRACES_DIR / "research_assistant")
    tracer.save_json(output_dir)
    print(f"Trace saved to {output_dir}/{trace.session_id}.json")
    print(f"Actions: {len(trace.actions)}, Escalations: {len(trace.escalations)}")


if __name__ == "__main__":
    run()
