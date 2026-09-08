
import httpx
import streamlit as st

from marketplace_agent.ui_preview import build_card_preview

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
                f"{API_URL}/demo/content",
                json={
                    "supplier_description": supplier_description,
                    "brand": brand,
                    "model": model,
                    "category": category,
                    "max_attempts": 1,
                },
                timeout=240.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            detail = ""
            if isinstance(error, httpx.HTTPStatusError):
                try:
                    detail = error.response.json().get("detail", "")
                except ValueError:
                    detail = ""
            message = detail or str(error) or type(error).__name__
            st.error(f"Pipeline недоступен: {message}")
        else:
            st.subheader("Результат GigaChat pipeline")
            result = response.json()
            st.session_state["last_pipeline_result"] = result
            st.metric("Статус", result["status"])
            st.metric("Попытки", result["attempts"])
            if result["violations"]:
                st.subheader("Нарушения")
                st.dataframe(result["violations"], hide_index=True)
            else:
                st.success("Нарушений не найдено")
            if result.get("attempt_history"):
                st.subheader("История попыток")
                st.dataframe(
                    result["attempt_history"],
                    hide_index=True,
                )
            st.json(result.get("content"))


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
    first, second, third = st.columns(3)
    first.metric("Recall таксономии", f"{taxonomy['overall_recall']:.1%}")
    second.metric("Purity кластеров", f"{clusters['weighted_purity']:.1%}")
    third.metric("Recall дефектов", f"{clusters['defect_recall']:.1%}")
    st.subheader("Слабые дефекты")
    st.write(taxonomy["weak_defects"] or "Нет")
    st.subheader("Ошибки таксономии")
    st.json(result["taxonomy_error_types"] or {})


def _render_support_chat() -> None:
    st.header("Чат поддержки")
    st.caption("Ответы основаны на правилах магазина и данных текущей сессии.")
    messages = st.session_state.setdefault("support_messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Напишите вопрос о доставке, возврате или заказе")
    if not question:
        return

    history = [
        {"role": item["role"], "content": item["content"]}
        for item in messages
    ]
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    try:
        response = httpx.post(
            f"{API_URL}/demo/support",
            json={
                "message": question,
                "history": history,
            },
            timeout=240.0,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as error:
        answer = f"Сервис поддержки недоступен: {error}"
        st.error(answer)
        return

    answer = payload["answer"]
    messages.append({"role": "assistant", "content": answer["text"]})
    st.session_state.setdefault("support_metrics", []).append(
        {
            "status": answer["status"],
            "latency_ms": payload["latency_ms"],
        }
    )
    with st.chat_message("assistant"):
        st.write(answer["text"])
        if answer["citations"]:
            st.caption("Источники: " + ", ".join(answer["citations"]))
        if answer["escalation_reason"]:
            st.warning("Вопрос передан специалисту.")


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
