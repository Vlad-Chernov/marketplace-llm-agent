from marketplace_agent.domain.models import GeneratedContent
from marketplace_agent.validation.deterministic import validate_deterministic

RULES = [
    {
        "id": "title-length",
        "description": "Заголовок должен содержать от 20 до 120 символов.",
        "field": "title",
        "min_length": 20,
        "max_length": 120,
        "severity": "medium",
    },
    {
        "id": "uppercase-ratio",
        "description": "Доля заглавных букв в заголовке не должна превышать 50 процентов.",
        "field": "title",
        "max_ratio": 0.5,
        "severity": "low",
    },
    {
        "id": "contact-information",
        "description": "Карточка не должна содержать телефон, email или внешний контакт.",
        "fields": ["title", "description"],
        "severity": "high",
    },
    {
        "id": "external-url",
        "description": "Карточка не должна содержать внешние ссылки.",
        "fields": ["title", "description"],
        "severity": "high",
    },
    {
        "id": "emoji-spam",
        "description": "Карточка не должна содержать более двух эмодзи.",
        "fields": ["title", "description"],
        "max_count": 2,
        "severity": "low",
    },
]


def make_content(title: str, description: str) -> GeneratedContent:
    return GeneratedContent(
        title=title,
        bullets=["Экран 14 дюймов", "Оперативная память 16 ГБ"],
        description=description,
        keywords=["Lenovo", "ноутбук"],
        used_attributes={"ram_gb": "16"},
    )


def test_accepts_safe_content() -> None:
    content = make_content(
        title="Lenovo IdeaPad 12345",
        description="Практичный ноутбук для повседневных задач.",
    )

    assert validate_deterministic(content, RULES) == []


def test_detects_all_seeded_deterministic_violations() -> None:
    content = make_content(
        title="СУПЕР НОУТБУК 😍😍😍",
        description=(
            "Звоните +7 999 123-45-67, пишите seller@example.com "
            "или переходите на https://example.com."
        ),
    )

    violations = validate_deterministic(content, RULES)

    assert {violation.rule_id for violation in violations} == {
        "title-length",
        "uppercase-ratio",
        "contact-information",
        "external-url",
        "emoji-spam",
    }


def test_accepts_title_at_minimum_length_boundary() -> None:
    content = make_content(
        title="Lenovo IdeaPad 12345",
        description="Описание без нарушений.",
    )

    violations = validate_deterministic(content, RULES)

    assert "title-length" not in {violation.rule_id for violation in violations}