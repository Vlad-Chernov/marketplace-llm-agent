# Evaluation Report

## Purpose

This document reports quality metrics for the golden evaluation set.

## Planned metrics

- Attribute extraction F1
- Hallucinated attribute rate
- Product-card rule compliance
- Defect detection recall
- Retrieval Recall@5
- Support-agent success rate
- Escalation precision
- Personal-data leak count
- Latency and cost

## Results

Evaluation results will be added after the relevant modules are implemented.
## Run history

| Run ID | Suite | Version | Cases | Errors | Average latency, ms | Cost, USD | Metrics |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| cac561167a2a475a8abf5e4ff486214a | mvp | baseline-attribute-v1 | 9 | 6 | 837 | 0.000000 | attribute_f1=0.051, hallucination_rate=0.800 |
| 05cacfe5224248d095d2a92d4ab30ac7 | mvp | baseline-attribute-openrouter-v1 | 9 | 0 | 9159 | 0.000000 | attribute_f1=0.043, hallucination_rate=0.818 |
| b6100c2c24654c1b9683cea268ceb00e | mvp | baseline-attribute-strict-v1 | 9 | 0 | 12238 | 0.000000 | attribute_f1=0.222, hallucination_rate=0.000 |
