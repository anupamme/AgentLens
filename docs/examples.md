# Trace Examples

## Example 1: Code Review Agent (`valid_trace_01.json`)

This trace captures a code review agent processing a GitHub pull request.

**Session overview:**
- Agent: `code-review-agent-v1`
- Task: Code review
- Outcome: Success
- Duration: ~5.5 minutes
- Actions: 7

**Action walkthrough:**

1. **Read** (full_auto) — Reads the PR diff. Fully autonomous since reading is low-risk.
2. **Reason** (full_auto) — Analyzes code changes and identifies 2 potential issues.
3. **Search** (auto_with_audit) — Searches codebase for similar patterns using `code_search` tool. Marked for audit since it accesses broader codebase.
4. **Reason** (full_auto) — Generates review comments with severity ratings.
5. **Communicate** (human_confirmed) — Posts inline comments on PR. Requires human confirmation before publishing.
6. **Write** (human_confirmed) — Drafts the summary review. Human confirms before submission.
7. **Communicate** (human_driven) — Final approval is human-driven; the agent assists but the human makes the decision.

**Key observations:**
- Autonomy decreases as actions become more externally visible
- Early analysis is fully autonomous; communication requires confirmation
- The final approval is human-driven — appropriate for consequential decisions
- Autonomy ratio: 4/7 (57%) of actions are autonomous

## Example 2: Research Assistant (`valid_trace_02.json`)

This trace shows a research assistant with a timeout and escalation event.

**Session overview:**
- Agent: `research-assistant-v2`
- Task: Research
- Outcome: Partial
- Duration: ~12.75 minutes
- Actions: 6
- Escalations: 1

**Action walkthrough:**

1. **Reason** (full_auto) — Decomposes the research query into sub-questions.
2. **Search** (full_auto) — Academic database search finds 12 relevant papers.
3. **Search** (auto_with_audit) — Web search **times out** after 30 seconds. This triggers an escalation.
4. **Reason** (full_auto) — Synthesizes findings from available papers despite the gap.
5. **Write** (human_confirmed) — Drafts a partial summary covering 2 of 3 sub-questions.
6. **Communicate** (human_driven) — Notifies user of partial completion.

**Escalation:**
- Triggered by action `act-103` (the timed-out web search)
- Reason: `confidence_low` — agent cannot verify recent findings
- Not resolved — the session ends with incomplete coverage

**Key observations:**
- The agent gracefully degrades when a tool fails (timeout)
- Escalation is recorded as a first-class event, not just a log message
- Session outcome is `partial` — the agent is honest about incomplete work
- Autonomy ratio: 3/6 (50%) — higher human involvement due to uncertainty
