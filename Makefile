.PHONY: setup test lint check data

setup:
	uv sync --all-groups

test:
	uv run pytest

lint:
	uv run ruff check --fix src tests
	uv run ruff check src tests

check: lint test

data:
	PYTHONPATH=src uv run python -m marketplace_agent.data_generation.generate

eval-smoke:
	PYTHONPATH=src uv run python scripts/llm_smoke.py