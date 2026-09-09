_DEFECT_NAMES_RU = {
    "battery_drain": "Быстрая разрядка батареи",
    "keyboard_failure": "Неисправность клавиатуры",
    "overheating": "Перегрев",
    "screen_flicker": "Мерцание экрана",
    "wifi_disconnect": "Обрывы Wi‑Fi",
}


def defect_display_name(defect_id: str) -> str:
    return _DEFECT_NAMES_RU.get(defect_id, defect_id)


def review_report_labels() -> dict[str, str]:
    """Return plain-language labels used by the review report UI."""
    return {
        "taxonomy_metric": "Точность классификации дефектов",
        "cluster_purity_metric": "Чистота тематических групп",
        "cluster_recall_metric": "Покрытие дефектов кластеризацией",
        "weak_defects": "Дефекты, которые классифицируются хуже всего",
        "taxonomy_errors": "Ошибки классификации",
        "distribution": "Распределение дефектов",
    }


def build_defect_distribution(
    defect_counts: dict[str, int],
) -> list[dict[str, object]]:
    total = sum(defect_counts.values())
    return [
        {
            "Дефект": defect_display_name(defect_id),
            "Отзывы": count,
            "Доля": count / total if total else 0.0,
        }
        for defect_id, count in sorted(
            defect_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
