from time import sleep

from fastapi.testclient import TestClient

from marketplace_agent import api
from marketplace_agent.api import app
from marketplace_agent.llm.providers import LLMProviderError
from marketplace_agent.support.agent import AgentAnswer


def test_content_demo_rejects_empty_supplier_description() -> None:
    response = TestClient(app).post(
        "/demo/content",
        json={"supplier_description": "", "brand": "Lenovo", "model": "IdeaPad"},
    )

    assert response.status_code == 422


def test_content_demo_exposes_provider_error(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "run_content_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            LLMProviderError("GigaChat authorization failed.")
        ),
    )

    response = TestClient(app, raise_server_exceptions=False).post(
        "/demo/content",
        json={
            "supplier_description": "Ноутбук Lenovo.",
            "brand": "Lenovo",
            "model": "IdeaPad",
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "GigaChat authorization failed."
    }


def test_review_report_returns_safe_metrics(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "_load_latest_review_report",
        lambda: {
            "run_id": "run-001",
            "created_at": "2026-09-06T18:27:35+00:00",
            "review_count": 200,
            "result": {
                "cleaned_review_count": 56,
                "taxonomy_metrics": {
                    "overall_recall": 1.0,
                    "weak_defects": [],
                },
                "cluster_metrics": {
                    "weighted_purity": 0.5,
                    "defect_recall": 0.0,
                    "cluster_count": 1,
                },
            },
        },
    )

    response = TestClient(app).get("/demo/reviews/report")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "run-001"
    assert payload["result"]["taxonomy_metrics"]["overall_recall"] == 1.0
    assert "review_text" not in response.text
    assert "defect_label" not in response.text


def test_support_endpoint_returns_answer_and_latency(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "_run_support_request",
        lambda _request: AgentAnswer(
            status="answered",
            text="Вернуть товар можно в течение 14 дней.",
            citations=["returns-01"],
        ),
    )

    response = TestClient(app).post(
        "/demo/support",
        json={"message": "Сколько дней можно вернуть товар?"},
    )

    assert response.status_code == 200
    assert response.json()["answer"]["status"] == "answered"
    assert response.json()["answer"]["citations"] == ["returns-01"]
    assert response.json()["latency_ms"] >= 0


def test_support_job_can_be_polled_until_completion(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "_run_support_request",
        lambda _request: AgentAnswer(
            status="answered",
            text="Ответ из фоновой задачи.",
            citations=["returns-01"],
        ),
    )

    client = TestClient(app)
    response = client.post(
        "/demo/support/jobs",
        json={"message": "Сколько дней можно вернуть товар?"},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    for _ in range(20):
        status = client.get(f"/demo/jobs/{job_id}")
        assert status.status_code == 200
        payload = status.json()
        if payload["status"] == "completed":
            assert payload["result"]["answer"]["text"] == (
                "Ответ из фоновой задачи."
            )
            break
        sleep(0.01)
    else:
        raise AssertionError("Background job did not complete.")
