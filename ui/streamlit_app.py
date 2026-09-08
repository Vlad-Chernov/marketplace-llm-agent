
import httpx
import streamlit as st

from marketplace_agent.ui_preview import build_card_preview

API_URL = "http://127.0.0.1:8000"


def main() -> None:
    st.set_page_config(page_title="Marketplace Agent", page_icon="🛒")
    st.title("Marketplace Agent")
    st.caption("Демо панели продавца: исходные данные и preview карточки")

    page = st.sidebar.radio("Раздел", ["Карточка товара", "Состояние API"])
    if page == "Состояние API":
        _render_health()
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
            st.error(f"Pipeline недоступен: {type(error).__name__}")
        else:
            st.subheader("Результат GigaChat pipeline")
            st.json(response.json())


if __name__ == "__main__":
    main()
