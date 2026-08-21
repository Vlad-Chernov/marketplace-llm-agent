# Read-only Support Tools: Design

## Goal

Provide a small, safe tool layer for the future support agent: product lookup, session-scoped order lookup, and policy search.

## Contract

Every tool returns `ToolResult` instead of raising expected errors:

- `ok: bool`;
- `data: dict | list | None`;
- `error: str | None`;
- `error_code: str | None`;
- `citations: list[str]`;
- `meta: dict`.

Success has `ok=True`, data, no error fields. Expected failure has `ok=False`, no data, and a stable error code.

## Tools

- `get_product(sku)`: reads a public product from `ProductRepository`; unknown SKU returns `not_found`.
- `get_order(order_id, session_id)`: reads an order only when its `session_id` matches; missing order returns `not_found`, a different owner returns `access_denied` without disclosing order data.
- `search_policy(query)`: calls the selected `hybrid-medium` retriever with `k=3`, returns chunk payloads and their `chunk_id` values as citations; blank queries return `invalid_arguments`.

All tools are read-only and catch expected repository, validation, and retrieval failures as `ToolResult` failures.

## Registry

`Tool` is a protocol with a name, description, input Pydantic model, and `run(**arguments) -> ToolResult`. `ToolRegistry` stores unique tools by name, validates arguments through the input model, calls the tool, and produces OpenAI-compatible function schemas from that same model. Tool names and argument contracts therefore exist in one place.

## Security boundary

Order ownership is enforced inside `get_order` now, not delegated to the later agent loop. The caller supplies `session_id` separately from user text. No tool writes data, executes code, or accepts arbitrary paths or SQL.

## Tests

Unit tests cover successful calls, unknown data, invalid arguments, ownership denial, policy citations, duplicate registry names, and generated tool schemas. They use a temporary SQLite database and a stub policy retriever.

## Out of scope

- LLM tool selection and multi-step agent loop;
- return eligibility calculation;
- mutations of orders, products, or policies.
