import json
from pathlib import Path

from marketplace_agent.evals.retrieval_metrics import RetrievalCase
from marketplace_agent.retrieval.documents import load_policy_chunks


def test_dataset_references_existing_policy_chunks() -> None:
    raw_cases = json.loads(
        Path("data/gold/policy_retrieval_cases.json").read_text(
            encoding="utf-8"
        )
    )
    cases = [RetrievalCase.model_validate(raw_case) for raw_case in raw_cases]
    chunk_ids = {
        chunk.chunk_id
        for chunk in load_policy_chunks(Path("data/support"))
    }

    assert len(cases) == 12
    assert {case.id for case in cases} == {
        f"policy-{index:03d}" for index in range(1, 13)
    }
    assert all(set(case.expected_chunk_ids) <= chunk_ids for case in cases)
    