# Policy Answers with Citations: Design

## Goal

Answer customer policy questions only from retrieved support-policy chunks and return traceable source identifiers. Return a safe `no_answer` when the knowledge base cannot support an answer.

## Scope

- Add `answer_policy_question(query, retriever, llm) -> PolicyAnswer`.
- Retrieve at most three policy chunks with the selected `hybrid-medium` retrieval path.
- Generate a short Russian answer from those chunks only.
- Validate citation identifiers in deterministic Python code.
- Add tests for a valid answer, missing retrieval context, and a fabricated citation.

## Data contract

`PolicyAnswer` has:

- `status`: `"answered"` or `"no_answer"`;
- `answer`: non-empty only for `"answered"`;
- `citations`: a list of policy `chunk_id` values, non-empty only for `"answered"`.

The LLM structured response has `answer` and `citations`. It cannot choose `status`; deterministic application code decides that status.

## Flow

1. Reject a blank query with `no_answer`.
2. Call `retriever.search(query, k=3)`.
3. If no chunks are returned, return `no_answer` without calling the LLM.
4. Send the question and only the returned chunks (`chunk_id`, heading, text) to the LLM in a strict structured-output prompt.
5. Accept the answer only when it has at least one citation and every cited `chunk_id` belongs to the retrieved set.
6. Return `no_answer` if structured generation fails, citations are missing, or any citation is unknown.

## Grounding boundary

The prompt explicitly forbids facts outside the passed chunks and instructs the LLM to say that information is unavailable if evidence is insufficient. Code can prove citation membership, not semantic entailment; later answer-quality evaluation will measure answer grounding separately.

## Error handling

Provider and structured-output failures are caught at the policy-answer boundary and converted to `no_answer`. Expected retrieval failures do not escape to the caller.

## Tests

Tests use `FakeLLMClient` and a stub retriever. They verify that the prompt contains retrieved evidence, valid citations are returned, a fabricated citation causes `no_answer`, and an empty retrieval result does not invoke the LLM.

## Out of scope

- Orders, products, and other support tools;
- tool selection or an autonomous agent;
- semantic verification beyond citation membership;
- response quality comparison across LLM models.
