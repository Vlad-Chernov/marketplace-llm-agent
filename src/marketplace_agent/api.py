from fastapi import FastAPI

app = FastAPI(
    title="Marketplace Agent API",
    version="0.1.0",
    description="HTTP API for marketplace agent workflows.",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight liveness response."""

    return {"status": "ok"}
