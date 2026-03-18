"""Code Review Agent — reads a diff, analyzes with Claude, posts findings.

Uses AnthropicBedrock client with AgentLens tracing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the src package is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config import ANTHROPIC_MODEL_ID, AWS_REGION, MAX_TOKENS_PER_CALL, TRACES_DIR

from agentlens.schema.enums import (
    ActionOutcome,
    ActionType,
    AutonomyLevel,
    SessionOutcome,
    TaskCategory,
)
from agentlens.sdk.tracer import AgentTracer

SAMPLE_DIFF = """\
--- a/server.py
+++ b/server.py
@@ -12,6 +12,10 @@ def handle_request(request):
     user_input = request.get("query")
-    result = db.execute(f"SELECT * FROM users WHERE name = '{user_input}'")
+    result = db.execute("SELECT * FROM users WHERE name = %s", (user_input,))
     return {"data": result}
+
+def health_check():
+    return {"status": "ok", "version": __version__}
"""


def run() -> None:
    tracer = AgentTracer(
        agent_type="code_reviewer",
        task_category=TaskCategory.CODE_REVIEW,
        model_used=ANTHROPIC_MODEL_ID,
    )
    tracer.start_session("Review pull request diff for security and style issues")

    # Step 1: Read the diff (FULL_AUTO)
    with tracer.action(
        action_type=ActionType.READ,
        autonomy_level=AutonomyLevel.FULL_AUTO,
        raw_input=SAMPLE_DIFF,
        tool_name="read_diff",
    ) as ctx:
        ctx.set_output_summary(
            "Read diff: 2 files changed, SQL injection fix + health check endpoint"
        )

    # Step 2: Analyze with Claude (FULL_AUTO)
    try:
        from anthropic import AnthropicBedrock

        client = AnthropicBedrock(aws_region=AWS_REGION)
        with tracer.action(
            action_type=ActionType.REASON,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            raw_input=f"Analyze this diff for issues:\n{SAMPLE_DIFF}",
            tool_name="claude_analyze",
        ) as ctx:
            response = client.messages.create(
                model=ANTHROPIC_MODEL_ID,
                max_tokens=MAX_TOKENS_PER_CALL,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Review this diff. List security issues, bugs, "
                            "and style problems.\n\n" + SAMPLE_DIFF
                        ),
                    }
                ],
            )
            analysis = response.content[0].text
            ctx.set_output_summary(analysis[:200])
    except Exception:
        # Fallback: simulate analysis when Bedrock is unavailable
        analysis = (
            "Finding 1: SQL injection fix looks correct — parameterized query.\n"
            "Finding 2: health_check uses __version__ but it is not imported.\n"
            "Finding 3: No input validation on request.get('query')."
        )
        tracer.record_action(
            action_type=ActionType.REASON,
            autonomy_level=AutonomyLevel.FULL_AUTO,
            outcome=ActionOutcome.SUCCESS,
            raw_input=f"Analyze this diff for issues:\n{SAMPLE_DIFF}",
            output_summary=analysis[:200],
            duration_ms=50,
            tool_name="claude_analyze_simulated",
        )

    # Step 3: Generate structured findings (FULL_AUTO)
    with tracer.action(
        action_type=ActionType.REASON,
        autonomy_level=AutonomyLevel.FULL_AUTO,
        raw_input=analysis,
    ) as ctx:
        findings = [
            {"severity": "info", "message": "SQL injection fix is correct"},
            {"severity": "error", "message": "__version__ not imported in health_check"},
            {"severity": "warning", "message": "Missing input validation on query param"},
        ]
        ctx.set_output_summary(f"Generated {len(findings)} findings")

    # Step 4: Post review comments (HUMAN_CONFIRMED)
    with tracer.action(
        action_type=ActionType.COMMUNICATE,
        autonomy_level=AutonomyLevel.HUMAN_CONFIRMED,
        raw_input=str(findings),
        tool_name="post_review_comments",
    ) as ctx:
        ctx.set_output_summary(f"Posted {len(findings)} review comments to PR")

    trace = tracer.end_session(SessionOutcome.SUCCESS, user_satisfaction_proxy=0.85)

    output_dir = str(TRACES_DIR / "code_reviewer")
    tracer.save_json(output_dir)
    print(f"Trace saved to {output_dir}/{trace.session_id}.json")
    print(f"Actions: {len(trace.actions)}, Autonomy ratio: {trace.autonomy_ratio:.0%}")


if __name__ == "__main__":
    run()
