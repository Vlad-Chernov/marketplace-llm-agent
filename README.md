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

## Архитектура

```text
Описание поставщика → извлечение атрибутов → генерация карточки → validation
Отзыв → классификация дефекта
Вопрос покупателя → guardrails → инструменты → поиск policy / заказ → ответ с цитатами
```

## Финальная оценка MVP

Запуск: `3a1a83ab1ea4463d9548c17915d413fe`<br>
Модель: `GigaChat-2-Pro:2.0.30.01`<br>
Набор: 25 golden-кейсов.

| Метрика | Значение | Порог MVP |
| --- | ---: | ---: |
| Технические ошибки | 0 | 0 |
| Attribute F1 | 1.000 | 1.000 |
| Validation accuracy | 1.000 | 1.000 |
| Review recall | 1.000 | 1.000 |
| Support tool accuracy | 1.000 | 1.000 |
| Latency p95 | 31 778 мс | менее 60 000 мс |

Подробности — в [evals/REPORT.md](evals/REPORT.md).

## Ограничения

- Golden-набор маленький; review recall измеряется на одном кейсе.
- Внешняя LLM может отвечать нестабильно по времени, поэтому клиент повторяет сетевые ошибки.
- Ресурсные метрики зависят от кэша.
- Стоимость сейчас равна `0.000000 USD`, потому что цены модели не указаны в `.env`; это не реальная стоимость API.
- Демо требует действующий ключ GigaChat и доступ к Hugging Face для embedding-модели.

## Проверки

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make check
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make eval
```

Сценарий показа проекта: [docs/DEMO.md](docs/DEMO.md).
