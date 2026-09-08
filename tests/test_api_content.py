from fastapi.testclient import TestClient

from marketplace_agent.api import app


def test_content_demo_rejects_empty_supplier_description() -> None:
    response = TestClient(app).post(
        "/demo/content",
        json={"supplier_description": "", "brand": "Lenovo", "model": "IdeaPad"},
    )

    assert response.status_code == 422
