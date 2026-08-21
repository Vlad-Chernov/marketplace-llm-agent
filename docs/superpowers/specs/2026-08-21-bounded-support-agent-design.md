# Bounded Support Agent: Design

## Goal

Create a minimal support-agent loop that uses only registered read-only tools, requests missing information, and escalates safely when it cannot complete a request.

## Contract

`SupportAgent.run(message, session_id, history) -> AgentAnswer` returns one of `answered`, `needs_clarification`, or `escalated`, with user-facing text, citations, and optional escalation reason.

The LLM returns structured `AgentDecision`: either `tool_call` with a registered tool name and JSON arguments, or `final` with text and an outcome. It cannot directly execute tools.

## Flow

1. Build the LLM prompt from the customer message, limited history, and schemas from `ToolRegistry`.
2. Parse one structured decision.
3. A final decision returns only `answered` or `needs_clarification`.
4. A tool decision is validated by `ToolRegistry`. Before calling `get_order`, the agent overwrites `arguments["session_id"]` with the separately supplied session ID.
5. Add the tool result to the next LLM turn and repeat, up to three tool calls.
6. Repeated tool name plus identical arguments, unknown tools, failed tool calls, malformed structured output, and the step limit return `escalated`.

## Safety

The model cannot set an effective session ID. The agent passes only registered tool calls through the registry. No write, shell, network, or arbitrary code tools exist. Tool errors are not exposed as internal tracebacks.

## Tests

Fake LLM decisions and stub tools test a cited policy answer, an owned order, a missing order number clarification, repeated calls, unknown tool calls, and step-limit escalation.

## Out of scope

- LLM retry strategies and long-term memory;
- write actions;
- production authentication beyond the supplied session ID.
