from decimal import Decimal
from pathlib import Path

import yaml

from marketplace_agent.catalog.extractor import extract_attributes
from marketplace_agent.content.pipeline import run_content_pipeline
from marketplace_agent.domain.models import (
    AttributeSpec,
    GeneratedContent,
    Product,
    Review,
)
from marketplace_agent.evals.llm_meter import MeteredLLMClient
from marketplace_agent.evals.models import GoldenCase
from marketplace_agent.evals.recording_registry import (
    RecordingToolRegistry,
)
from marketplace_agent.evals.runner import EvaluationPrediction
from marketplace_agent.reviews.analyzer import classify_review
from marketplace_agent.reviews.taxonomy import DefectTaxonomy
from marketplace_agent.support.agent import (
    SupportAgent,
)
from marketplace_agent.support.agent import (
    ToolRegistry as SupportToolRegistry,
)
from marketplace_agent.validation.deterministic import validate_deterministic
from marketplace_agent.validation.semantic import validate_semantic


class MvpCaseExecutor:
    """Запускает реальную логику приложения для одного eval-кейса."""

    def __init__(
        self,
        llm: MeteredLLMClient,
        attribute_specs: list[AttributeSpec],
        defect_taxonomy: DefectTaxonomy | None = None,
        support_registry: SupportToolRegistry | None = None,
    ) -> None:
        self._llm = llm
        self._attribute_specs = attribute_specs
        self._defect_taxonomy = defect_taxonomy
        self._support_registry = support_registry

    def execute(self, case: GoldenCase) -> EvaluationPrediction:
        """Выполнить один eval-кейс."""

        self._llm.reset()

        if case.type == "attribute_extraction":
            return self._execute_attribute_extraction(case)

        if case.type == "validation":
            return self._execute_validation(case)

        if case.type == "review_analysis":
            return self._execute_review_analysis(case)
        if case.type in {"support", "no_answer", "adversarial"}:
            return self._execute_support_case(case)
        if case.type == "content_generation":
            return self._execute_content_generation(case)
        raise ValueError(f"Unsupported case type: {case.type}")

    def _execute_attribute_extraction(
        self,
        case: GoldenCase,
    ) -> EvaluationPrediction:
        product = Product(
            sku="LAP-EVAL-001",
            category="laptops",
            brand="Evaluation",
            model="Case",
            price=Decimal(1),
            sales_count=0,
            supplier_description=case.input["supplier_description"],
        )
        result = extract_attributes(
            product=product,
            attribute_specs=self._attribute_specs,
            client=self._llm,
        )

        response = self._llm.last_response
        if response is None:
            raise RuntimeError("Attribute extractor did not call the LLM.")

        return EvaluationPrediction(
            prediction={
                key: attribute.value
                for key, attribute in result.attributes.items()
            },
            raw_response=response.content,
            model=response.model,
            latency_ms=self._llm.latency_ms,
            prompt_tokens=self._llm.prompt_tokens,
            completion_tokens=self._llm.completion_tokens,
            cost_usd=self._llm.cost_usd,
        )

    def _execute_validation(
        self,
        case: GoldenCase,
        ) -> EvaluationPrediction:
        content = GeneratedContent.model_validate(case.input["content"])
        deterministic_rules, semantic_rules = _load_validation_rules()

        violations = validate_deterministic(content, deterministic_rules)
        violations.extend(
            validate_semantic(content, semantic_rules, self._llm)
        )

        response = self._llm.last_response
        if response is None:
            raise RuntimeError("Semantic validator did not call the LLM.")

        return EvaluationPrediction(
            prediction={
                "violations": [
                    violation.rule_id for violation in violations
                ]
            },
            raw_response=response.content,
            model=response.model,
            latency_ms=self._llm.latency_ms,
            prompt_tokens=self._llm.prompt_tokens,
            completion_tokens=self._llm.completion_tokens,
            cost_usd=self._llm.cost_usd,
        )

    def _execute_content_generation(
        self,
        case: GoldenCase,
    ) -> EvaluationPrediction:
        product = Product.model_validate(case.input["product"])
        result = run_content_pipeline(product, self._llm)

        response = self._llm.last_response
        if response is None:
            raise RuntimeError("Content pipeline did not call the LLM.")

        content = result.content
        return EvaluationPrediction(
            prediction={
                "status": result.status,
                "title": None if content is None else content.title,
                "used_attributes": (
                    {} if content is None else content.used_attributes
                ),
            },
            raw_response=response.content,
            model=response.model,
            latency_ms=self._llm.latency_ms,
            prompt_tokens=self._llm.prompt_tokens,
            completion_tokens=self._llm.completion_tokens,
            cost_usd=self._llm.cost_usd,
        )
    
    def _execute_review_analysis(
        self,
        case: GoldenCase,
    ) -> EvaluationPrediction:
        if self._defect_taxonomy is None:
            raise RuntimeError("Review analysis requires a defect taxonomy.")

        review = Review.model_validate(case.input["review"])
        labels = classify_review(
            review,
            self._defect_taxonomy,
            self._llm,
        )

        response = self._llm.last_response
        if response is None:
            raise RuntimeError("Review analyzer did not call the LLM.")

        return EvaluationPrediction(
            prediction={"defect_label": labels.defect_id},
            raw_response=response.content,
            model=response.model,
            latency_ms=self._llm.latency_ms,
            prompt_tokens=self._llm.prompt_tokens,
            completion_tokens=self._llm.completion_tokens,
            cost_usd=self._llm.cost_usd,
        )

    def _execute_support_case(
        self,
        case: GoldenCase,
    ) -> EvaluationPrediction:
        if self._support_registry is None:
            raise RuntimeError("Support case requires a tool registry.")

        if isinstance(
            self._support_registry,
            RecordingToolRegistry,
        ):
            self._support_registry.reset()

        answer = SupportAgent(
            registry=self._support_registry,
            llm=self._llm,
        ).run(
            message=case.input["message"],
            session_id=case.input["session_id"],
            history=[],
        )

        response = self._llm.last_response
        tool_calls = (
            list(self._support_registry.called_tools)
            if isinstance(
                self._support_registry,
                RecordingToolRegistry,
            )
            else []
        )

        return EvaluationPrediction(
            prediction={
                "status": answer.status,
                "text": answer.text,
                "citations": answer.citations,
            },
            raw_response="" if response is None else response.content,
            model="guardrail" if response is None else response.model,
            latency_ms=self._llm.latency_ms,
            prompt_tokens=self._llm.prompt_tokens,
            completion_tokens=self._llm.completion_tokens,
            cost_usd=self._llm.cost_usd,
            tool_calls=tool_calls,
        )

def _load_validation_rules() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
]:
    rules_path = (
        Path(__file__).resolve().parents[3]
        / "data"
        / "processed"
        / "rules.yaml"
    )
    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))

    return (
        list(rules["deterministic_rules"]),
        list(rules["semantic_rules"]),
    )