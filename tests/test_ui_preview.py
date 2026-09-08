from marketplace_agent.ui_preview import build_card_preview


def test_builds_card_preview_without_llm_call() -> None:
    preview = build_card_preview(
        "Ноутбук для работы",
        brand="Lenovo",
        model="IdeaPad",
        category="laptops",
    )

    assert preview.title == "Lenovo IdeaPad"
    assert preview.description == "Ноутбук для работы"
    assert "Категория: laptops" in preview.bullets
