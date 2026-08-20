from pathlib import Path

import pytest

from marketplace_agent.retrieval.documents import load_policy_chunks

SUPPORT_PATH = Path("data/support")


def test_loads_citable_chunks_from_support_policies() -> None:
    chunks = load_policy_chunks(SUPPORT_PATH)

    assert len(chunks) == 12
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert {chunk.document_id for chunk in chunks} == {
        "delivery",
        "exchange",
        "returns",
        "warranty",
    }
    assert all(chunk.heading for chunk in chunks)
    assert all(chunk.text for chunk in chunks)

    returns_chunks = [
        chunk for chunk in chunks if chunk.document_id == "returns"
    ]
    assert [chunk.chunk_id for chunk in returns_chunks] == [
        "returns-01",
        "returns-02",
        "returns-03",
    ]
    assert returns_chunks[0].heading == "Возврат товара"
    assert "14 календарных дней" in returns_chunks[0].text


def test_splits_document_with_second_level_headings(tmp_path: Path) -> None:
    policy_path = tmp_path / "sample.md"
    policy_path.write_text(
        "# Тестовая политика\n\n"
        "## Первый пункт\n\n"
        "Первое правило.\n\n"
        "## Второй пункт\n\n"
        "Второе правило.\n",
        encoding="utf-8",
    )

    chunks = load_policy_chunks(tmp_path)

    assert [(chunk.chunk_id, chunk.heading, chunk.text) for chunk in chunks] == [
        ("sample-01", "Первый пункт", "Первое правило."),
        ("sample-02", "Второй пункт", "Второе правило."),
    ]


def test_rejects_policy_without_non_empty_text(tmp_path: Path) -> None:
    (tmp_path / "empty.md").write_text(
        "# Пустая политика\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no non-empty policy text"):
        load_policy_chunks(tmp_path)