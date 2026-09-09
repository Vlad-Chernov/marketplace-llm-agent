
import httpx
import streamlit as st

from marketplace_agent.ui_preview import build_card_preview
from marketplace_agent.ui_review_report import (
    build_defect_distribution,
    defect_display_name,
    review_report_labels,
)

API_URL = "http://127.0.0.1:8000"


def main() -> None:
    st.set_page_config(page_title="Marketplace Agent", page_icon="🛒")
    st.title("Marketplace Agent")
    st.caption("Демо панели продавца: исходные данные и preview карточки")

    page = st.sidebar.radio(
        "Раздел",
        [
            "Карточка товара",
            "Отчёт по отзывам",
            "Чат поддержки",
            "Метрики",
            "Состояние API",
        ],
    )
    if page == "Состояние API":
        _render_health()
        return
    if page == "Отчёт по отзывам":
        _render_review_report()
        return
    if page == "Чат поддержки":
        _render_support_chat()
        return
    if page == "Метрики":
        _render_metrics()
        return

    _render_card()


def _render_health() -> None:
    st.header("Состояние backend")
    try:
        response = httpx.get(f"{API_URL}/health", timeout=3.0)
        response.raise_for_status()
    except httpx.HTTPError as error:
        st.error(f"FastAPI недоступен: {type(error).__name__}")
        return
    st.success("FastAPI работает")
    st.json(response.json())


def _render_card() -> None:
    st.header("Карточка товара")
    with st.form("card-preview"):
        supplier_description = st.text_area(
            "Описание поставщика",
            value="Ноутбук для работы и учёбы.",
            height=120,
        )
        brand = st.text_input("Бренд", value="Lenovo")
        model = st.text_input("Модель", value="IdeaPad")
        category = st.text_input("Категория", value="laptops")
        show_preview = st.form_submit_button("Показать preview")
        generate = st.form_submit_button("Сгенерировать через GigaChat")

    if not show_preview and not generate:
        content_job_id = st.session_state.get("content_job_id")
        if content_job_id:
            _poll_content_job(content_job_id)
        elif st.session_state.get("last_pipeline_result"):
            _show_pipeline_result(st.session_state["last_pipeline_result"])
        elif st.session_state.get("content_job_error"):
            st.error(st.session_state["content_job_error"])
        return

    preview = build_card_preview(
        supplier_description,
        brand=brand,
        model=model,
        category=category,
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Исходные данные")
        st.code(supplier_description or "(пусто)")
    with right:
        st.subheader("Preview карточки")
        st.write(f"**{preview.title}**")
        st.write(preview.description)
        for bullet in preview.bullets:
            st.markdown(f"- {bullet}")

    if generate:
        try:
            response = httpx.post(
                f"{API_URL}/demo/content/jobs",
                json={
                    "supplier_description": supplier_description,
                    "brand": brand,
                    "model": model,
                    "category": category,
                    "max_attempts": 1,
                },
                timeout=5.0,
            )
            response.raise_for_status()
            st.session_state["content_job_id"] = response.json()["job_id"]
            st.session_state["content_job_error"] = None
        except httpx.HTTPError as error:
            st.error(f"Не удалось запустить pipeline: {error}")

    content_job_id = st.session_state.get("content_job_id")
    if content_job_id:
        _poll_content_job(content_job_id)
    elif st.session_state.get("last_pipeline_result"):
        _show_pipeline_result(st.session_state["last_pipeline_result"])
    elif st.session_state.get("content_job_error"):
        st.error(st.session_state["content_job_error"])


def _show_pipeline_result(result: dict[str, object]) -> None:
    st.subheader("Результат GigaChat pipeline")
    st.metric("Статус", result["status"])
    st.metric("Попытки", result["attempts"])
    if result["violations"]:
        st.subheader("Нарушения")
        st.dataframe(result["violations"], hide_index=True)
    else:
        st.success("Нарушений не найдено")
    if result.get("attempt_history"):
        st.subheader("История попыток")
        st.dataframe(result["attempt_history"], hide_index=True)
    st.json(result.get("content"))


@st.fragment(run_every=2)
def _poll_content_job(job_id: str) -> None:
    try:
        response = httpx.get(f"{API_URL}/demo/jobs/{job_id}", timeout=5.0)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as error:
        st.error(f"Статус pipeline недоступен: {error}")
        return

    if payload["status"] in {"queued", "running"}:
        st.info("Pipeline выполняется в фоне. Можно переходить по вкладкам.")
    elif payload["status"] == "completed":
        st.session_state["content_job_id"] = None
        st.session_state["content_job_error"] = None
        st.session_state["last_pipeline_result"] = payload["result"]
        _show_pipeline_result(payload["result"])
    else:
        st.session_state["content_job_id"] = None
        error = f"Pipeline завершился с ошибкой: {payload.get('error')}"
        st.session_state["content_job_error"] = error
        st.error(error)


def _render_review_report() -> None:
    st.header("Отчёт по отзывам")
    try:
        response = httpx.get(
            f"{API_URL}/demo/reviews/report",
            timeout=5.0,
        )
        response.raise_for_status()
        report = response.json()
    except httpx.HTTPError as error:
        st.error(f"Отчёт недоступен: {error}")
        return

    if report["status"] != "available":
        st.info(report["message"])
        return

    result = report["result"]
    st.caption(
        f"Запуск {report['run_id']} · отзывов: {report['review_count']} · "
        f"очищено: {result['cleaned_review_count']}"
    )
    taxonomy = result["taxonomy_metrics"]
    clusters = result["cluster_metrics"]
    labels = review_report_labels()
    st.subheader("Классификация отзывов по типам дефектов")
    st.caption(
        "LLM сопоставляет каждый отзыв с фиксированной категорией дефекта."
    )
    first, second, third = st.columns(3)
    first.metric(labels["taxonomy_metric"], f"{taxonomy['overall_recall']:.1%}")
    st.subheader("Эксперимент: объединение похожих отзывов")
    st.caption(
        "Эмбеддинги группируются без таксономии; метрики нужны для проверки "
        "качества таких групп."
    )
    second.metric(
        labels["cluster_purity_metric"],
        f"{clusters['weighted_purity']:.1%}",
    )
    third.metric(
        labels["cluster_recall_metric"],
        f"{clusters['defect_recall']:.1%}",
    )
    st.subheader(labels["weak_defects"])
    st.write(
        [defect_display_name(item) for item in taxonomy["weak_defects"]]
        or "Нет"
    )
    st.subheader(labels["distribution"])
    distribution = build_defect_distribution(
        taxonomy.get("defect_counts", {})
    )
    if distribution:
        st.dataframe(
            [
                {
                    **row,
                    "Доля": f"{row['Доля']:.1%}",
                }
                for row in distribution
            ],
            hide_index=True,
        )
    else:
        st.write("Дефекты не обнаружены")
    st.subheader(labels["taxonomy_errors"])
    st.json(result["taxonomy_error_types"] or {})


def _render_support_chat() -> None:
    st.header("Чат поддержки")
    st.caption("Ответы основаны на правилах магазина и данных текущей сессии.")
    messages = st.session_state.setdefault("support_messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("escalation_reason"):
                st.warning(
                    "Причина эскалации: "
                    f"{message['escalation_reason']}"
                )

    support_job_id = st.session_state.get("support_job_id")
    if support_job_id:
        _poll_support_job(support_job_id)

    question = st.chat_input(
        "Напишите вопрос о доставке, возврате или заказе",
        disabled=bool(support_job_id),
    )
    if not question:
        return

    history = [
        {"role": item["role"], "content": item["content"]}
        for item in messages
    ]
    try:
        response = httpx.post(
            f"{API_URL}/demo/support/jobs",
            json={
                "message": question,
                "history": history,
            },
            timeout=5.0,
        )
        response.raise_for_status()
        st.session_state["support_job_id"] = response.json()["job_id"]
        st.session_state["support_pending_question"] = question
    except httpx.HTTPError as error:
        answer = f"Сервис поддержки недоступен: {error}"
        st.error(answer)
        return

    st.rerun()


@st.fragment(run_every=2)
def _poll_support_job(job_id: str) -> None:
    try:
        response = httpx.get(f"{API_URL}/demo/jobs/{job_id}", timeout=5.0)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as error:
        st.error(f"Статус поддержки недоступен: {error}")
        return

    if payload["status"] in {"queued", "running"}:
        st.info("Ответ готовится в фоне. Можно переходить по вкладкам.")
        return
    if payload["status"] != "completed":
        st.session_state["support_job_id"] = None
        st.error(f"Поддержка завершилась с ошибкой: {payload.get('error')}")
        return

    result = payload["result"]
    messages = st.session_state.setdefault("support_messages", [])
    messages.append(
        {
            "role": "user",
            "content": st.session_state.pop("support_pending_question", ""),
        }
    )
    answer = result["answer"]
    messages.append(
        {
            "role": "assistant",
            "content": answer["text"],
            "escalation_reason": answer["escalation_reason"],
        }
    )
    st.session_state.setdefault("support_metrics", []).append(
        {
            "status": answer["status"],
            "latency_ms": result["latency_ms"],
        }
    )
    st.session_state["support_job_id"] = None
    st.rerun()


def _render_metrics() -> None:
    st.header("Метрики демо")
    pipeline_result = st.session_state.get("last_pipeline_result")
    support_metrics = st.session_state.get("support_metrics", [])
    first, second, third = st.columns(3)
    first.metric(
        "Последний статус карточки",
        pipeline_result["status"] if pipeline_result else "—",
    )
    second.metric(
        "Нарушения карточки",
        len(pipeline_result["violations"]) if pipeline_result else 0,
    )
    third.metric("Вопросов поддержки", len(support_metrics))
    if support_metrics:
        st.subheader("Запросы поддержки")
        st.dataframe(support_metrics, hide_index=True)


if __name__ == "__main__":
    main()
