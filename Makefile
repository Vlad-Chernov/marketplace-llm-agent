.PHONY: setup test lint check data ingest

setup:
	uv sync --all-groups

test:
	uv run pytest

lint:
	uv run ruff check --fix src tests evals/run_eval.py evals/compare_content_pipeline.py evals/run_review_eval.py scripts/ingest_policies.py
	uv run ruff check src tests evals/run_eval.py evals/compare_content_pipeline.py evals/run_review_eval.py scripts/ingest_policies.py

check: lint test

data:
	PYTHONPATH=src uv run python -m marketplace_agent.data_generation.generate

ingest:
	uv run python scripts/ingest_policies.py

eval-smoke:
	PYTHONPATH=src uv run python scripts/llm_smoke.py