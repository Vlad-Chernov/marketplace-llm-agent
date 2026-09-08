from dataclasses import dataclass


@dataclass(frozen=True)
class CardPreview:
    title: str
    description: str
    bullets: tuple[str, ...]


def build_card_preview(
    supplier_description: str,
    *,
    brand: str,
    model: str,
    category: str,
) -> CardPreview:
    """Build a deterministic UI preview without making an LLM request."""

    title = " ".join(part for part in (brand, model) if part).strip()
    return CardPreview(
        title=title or "Карточка товара",
        description=supplier_description.strip(),
        bullets=tuple(
            part
            for part in (
                f"Категория: {category}" if category else "",
                f"Бренд: {brand}" if brand else "",
                f"Модель: {model}" if model else "",
            )
            if part
        ),
    )
