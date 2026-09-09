# Marketplace LLM Agent

Учебный MVP LLM-агента для маркетплейса: создаёт карточку товара, анализирует отзыв и отвечает покупателю по правилам магазина.

## Возможности

- извлечение характеристик из шумного описания поставщика;
- генерация и проверка карточки ноутбука;
- классификация дефектов в отзывах;
- поддержка с поиском по правилам и цитатами;
- golden evaluation из 25 кейсов;
- кэширование LLM-ответов для повторных прогонов.

## Стек

Python, uv, Pydantic, SQLite, BM25, векторный поиск, pytest, Ruff, GigaChat API.

## Быстрый запуск

Нужны Python, `uv` и ключ авторизации GigaChat.

```bash
git clone <URL_ТВОЕГО_РЕПОЗИТОРИЯ>
cd <ПАПКА_ПРОЕКТА>

cp .env.example .env
```

В `.env` укажи:

```ini
LLM_PROVIDER=gigachat
GIGACHAT_AUTHORIZATION_KEY=<твой_ключ>
GIGACHAT_MODEL=GigaChat-2-Pro
LLM_INPUT_PRICE_PER_MILLION=0.0
LLM_OUTPUT_PRICE_PER_MILLION=0.0
```

Установи зависимости:

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make setup
```

Для GigaChat на macOS добавь сертификат Минцифры в certifi текущего окружения:

```bash
GIGACHAT_CERTIFI_BUNDLE=$(UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache \
  uv run python -m certifi)

curl -k "https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt" \
  -w "\n" >> "$GIGACHAT_CERTIFI_BUNDLE"
```

Если в терминале ранее был задан другой провайдер, очисти его:

```bash
unset LLM_PROVIDER
```

Подготовь данные и индекс правил:

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make data
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make ingest
```

Запусти демонстрацию:

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make demo
```

Полная оценка MVP:

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make eval
```

Чтобы видеть длительность LLM-запросов и примерное оставшееся время
оценки, включи прогресс только для одной команды:

```bash
LLM_PROGRESS=1 UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make eval
```

Для обычного тихого запуска оставь `LLM_PROGRESS=0`.

## Архитектура

```text
Описание поставщика → извлечение атрибутов → генерация карточки → validation
Отзыв → классификация дефекта
Вопрос покупателя → guardrails → инструменты → поиск policy / заказ → ответ с цитатами
```

## Локальная веб-демонстрация

Запусти FastAPI и Streamlit в двух терминалах:

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make api
```

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make ui
```

Открой `http://localhost:8501`. В интерфейсе доступны карточка товара,
отчёт по отзывам, чат поддержки, метрики и состояние API.

Генерация карточки и ответы поддержки запускаются как фоновые задачи.
После нажатия кнопки можно переходить между вкладками: Streamlit проверяет
статус задачи автоматически и показывает результат после завершения.

API-документация доступна по адресу `http://127.0.0.1:8000/docs`.

## Финальная оценка MVP

Запуск: `df2607258c6146a78766ad8ab80461e8`<br>
Модель: `GigaChat-2-Pro`<br>
Набор: 50 golden-кейсов.

| Метрика | Значение | Порог MVP |
| --- | ---: | ---: |
| Технические ошибки | 0 | 0 |
| Attribute F1 | 0.985 | не менее 0.950 |
| Validation accuracy | 1.000 | 1.000 |
| Review recall | 1.000 | 1.000 |
| Support tool accuracy | 0.812 | не менее 0.800 |

Подробности — в [evals/REPORT.md](evals/REPORT.md).

## Ограничения

- Golden-набор синтетический и содержит 50 кейсов; результаты нужно отдельно
  подтвердить на реальных данных.
- Внешняя LLM может отвечать нестабильно по времени, поэтому клиент повторяет сетевые ошибки.
- Финальный запуск полностью обслужен кэшем, поэтому его нулевая задержка и
  стоимость не характеризуют внешний API.
- По измеренному расходу LangGraph оценка генерации составляет около 2 020 ₽
  на 1 000 товаров и 202 017 ₽ на 100 000 товаров при тарифе 500 ₽ за 1 млн
  токенов GigaChat 2 Pro.
- Демо требует действующий ключ GigaChat и доступ к Hugging Face для embedding-модели.
- Фоновые задачи хранятся в памяти процесса FastAPI; перезапуск backend удаляет
  незавершённые demo-задачи.

## Проверки

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make check
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make eval
```

Сценарий показа проекта: [docs/DEMO.md](docs/DEMO.md).
