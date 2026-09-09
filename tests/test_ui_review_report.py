from marketplace_agent.ui_review_report import (
    build_defect_distribution,
    defect_display_name,
    review_report_labels,
)


def test_uses_plain_language_for_review_report_metrics() -> None:
    labels = review_report_labels()

    assert labels == {
        "taxonomy_metric": "Точность классификации дефектов",
        "cluster_purity_metric": "Чистота тематических групп",
        "cluster_recall_metric": "Покрытие дефектов кластеризацией",
        "weak_defects": "Дефекты, которые классифицируются хуже всего",
        "taxonomy_errors": "Ошибки классификации",
        "distribution": "Распределение дефектов",
    }


def test_builds_defect_distribution_as_percentages() -> None:
    assert build_defect_distribution(
        {"overheating": 2, "battery_drain": 1}
    ) == [
        {"Дефект": "Перегрев", "Отзывы": 2, "Доля": 2 / 3},
        {
            "Дефект": "Быстрая разрядка батареи",
            "Отзывы": 1,
            "Доля": 1 / 3,
        },
    ]


def test_translates_defect_ids_for_display() -> None:
    assert defect_display_name("battery_drain") == "Быстрая разрядка батареи"
    assert defect_display_name("unknown_defect") == "unknown_defect"
