.PHONY: setup test lint check data ingest demo evaluate-retrieval compare-retrieval evaluate-mvp

setup:
	uv sync --all-groups

test:
	uv run pytest

lint:
	uv run ruff check --fix src tests evals/run_eval.py evals/compare_content_pipeline.py evals/run_review_eval.py scripts/ingest_policies.py evals/evaluate_retrieval.py evals/compare_retrieval_experiments.py scripts/run_mvp_final.py scripts/summarize_mvp_run.py
	uv run ruff check src tests evals/run_eval.py evals/compare_content_pipeline.py evals/run_review_eval.py scripts/ingest_policies.py evals/evaluate_retrieval.py evals/compare_retrieval_experiments.py scripts/run_mvp_final.py scripts/summarize_mvp_run.py

check: lint test

data:
	PYTHONPATH=src uv run python -m marketplace_agent.data_generation.generate

ingest:
	uv run python scripts/ingest_policies.py

demo:
	uv run python scripts/demo.py

eval-smoke:
	PYTHONPATH=src uv run python scripts/llm_smoke.py

evaluate-retrieval:
	uv run python evals/evaluate_retrieval.py

compare-retrieval:
	uv run python evals/compare_retrieval_experiments.py

evaluate-mvp:
	uv run python scripts/run_mvp_final.py