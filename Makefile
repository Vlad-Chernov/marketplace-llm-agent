.PHONY: setup test lint check

setup:
	uv sync --all-groups

test:
	uv run pytest

lint:
	uv run ruff check src tests

check: test lint