.PHONY: setup test lint check data

setup:
	uv sync --all-groups

test:
	uv run pytest

lint:
	uv run ruff check src tests

check: test lint

data:
	PYTHONPATH=src uv run python -m marketplace_agent.data_generation.generate