import json
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import BaseModel

from marketplace_agent import api
from marketplace_agent.api import app


def test_health_endpoint() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_background_job_writes_trace(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class Result(BaseModel):
        value: str

    monkeypatch.setattr(api, "_project_root", lambda: tmp_path)
    api._jobs["job-001"] = {
        "job_id": "job-001",
        "kind": "content",
        "status": "queued",
        "result": None,
        "error": None,
    }

    api._execute_job("job-001", lambda: Result(value="ok"))

    trace_path = tmp_path / "data" / "traces" / "content-job-001.jsonl"
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert api._jobs["job-001"]["status"] == "completed"
    assert [event["event_type"] for event in events] == [
        "run_started",
        "job_started",
        "job_completed",
        "run_completed",
    ]
