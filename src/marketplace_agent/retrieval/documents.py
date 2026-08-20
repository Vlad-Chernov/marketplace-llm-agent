from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class PolicyChunk(BaseModel):
    """Store one source fragment for retrieval and citation."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    policy_type: str = Field(min_length=1)
    heading: str = Field(min_length=1)
    text: str = Field(min_length=1)


def load_policy_chunks(path: Path | str) -> list[PolicyChunk]:
    """Load support Markdown documents as structured citation chunks."""

    directory = Path(path)
    chunks: list[PolicyChunk] = []
    document_ids: set[str] = set()

    for document_path in sorted(directory.glob("*.md")):
        document_id = document_path.stem
        if document_id in document_ids:
            raise ValueError(f"Duplicate document ID: {document_id}")
        document_ids.add(document_id)

        chunks.extend(
            _parse_document(
                document_path.read_text(encoding="utf-8"),
                document_id,
            )
        )

    if not chunks:
        raise ValueError("No policy chunks found.")

    return chunks


def _parse_document(content: str, document_id: str) -> list[PolicyChunk]:
    lines = content.splitlines()
    document_heading = _read_document_heading(lines)
    sections = _split_by_second_level_headings(lines)

    if sections:
        parts = sections
    else:
        parts = [
            (document_heading, paragraph)
            for paragraph in _read_paragraphs_after_heading(lines)
        ]

    if not parts:
        raise ValueError(f"{document_id}: no non-empty policy text")

    return [
        PolicyChunk(
            chunk_id=f"{document_id}-{index:02d}",
            document_id=document_id,
            policy_type=document_id,
            heading=heading,
            text=text,
        )
        for index, (heading, text) in enumerate(parts, start=1)
    ]


def _read_document_heading(lines: list[str]) -> str:
    for line in lines:
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    raise ValueError("Document heading is required.")


def _split_by_second_level_headings(
    lines: list[str],
) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading: str | None = None
    text_lines: list[str] = []

    for line in lines:
        if line.startswith("## "):
            if heading is not None:
                text = "\n".join(text_lines).strip()
                if not text:
                    raise ValueError(f"{heading}: no non-empty policy text")
                sections.append((heading, text))
            heading = line.removeprefix("## ").strip()
            text_lines = []
        elif heading is not None:
            text_lines.append(line)

    if heading is not None:
        text = "\n".join(text_lines).strip()
        if not text:
            raise ValueError(f"{heading}: no non-empty policy text")
        sections.append((heading, text))

    return sections


def _read_paragraphs_after_heading(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    paragraph_lines: list[str] = []
    after_heading = False

    for line in lines:
        if line.startswith("# "):
            after_heading = True
            continue
        if not after_heading:
            continue
        if line.strip():
            paragraph_lines.append(line.strip())
        elif paragraph_lines:
            paragraphs.append(" ".join(paragraph_lines))
            paragraph_lines = []

    if paragraph_lines:
        paragraphs.append(" ".join(paragraph_lines))

    return paragraphs