# Performance & Grounding Analysis Tables

### Table 1: Load Shape Comparison Across Traffic Scenarios

| Scenario | Total Reqs | Throughput (req/s) | p50 TTFT (ms) | p95 TTFT (ms) | p99 TTFT (ms) | Error Rate | Fallback Rate | Est. Cost (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `steady` | 42 | 0.17 | 15.5 | 5069.1 | 5075.8 | 4.8% | 23.8% | $0.00291 |
| `burst` | 30 | 2.56 | 0.0 | 12.4 | 13.6 | 0.0% | 40.0% | $0.00014 |
| `sustained_overload` | 26 | 2.76 | 0.0 | 0.0 | 0.0 | 0.0% | 30.8% | $0.00007 |
| `rpm_halved` | 12 | 1.31 | 0.0 | 0.0 | 0.0 | 0.0% | 25.0% | $0.00003 |
| `daily_exhausted` | 12 | 1.63 | 0.0 | 0.0 | 0.0 | 0.0% | 25.0% | $0.00003 |

### Table 2a: Cost & Latency by Retrieval Setting

| Retrieval Setting | N | Mean Prompt Tokens | Mean Output Tokens | p50 TTFT (ms) | p95 TTFT (ms) | p50 Total (ms) | Cost / Req (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `setting_a` | 93 | 125 | 24 | 0.0 | 13.6 | 0.0 | $0.000008 |
| `setting_b` | 15 | 1129 | 289 | 32.3 | 5075.6 | 818.1 | $0.000080 |
| `setting_c` | 14 | 1175 | 376 | 5064.8 | 5075.8 | 5057.0 | $0.000089 |

### Table 2b: Retrieval Quality & Grounding by Setting

| Retrieval Setting | N | Correct Citations | Correct Refusals | Hallucinations | False Refusals | Overall Grounding Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `setting_a` | 93 | 8 | 30 | 0 | 55 | 40.9% |
| `setting_b` | 15 | 8 | 2 | 4 | 1 | 66.7% |
| `setting_c` | 14 | 8 | 2 | 3 | 1 | 71.4% |

### Table 3: Fallback and Degradation Behavior Under Stress

| Scenario | Total Invocations | Direct LLM (OK) | Rate-Limited Fallbacks | Unhandled Errors | Total Retries Executed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `steady` | 42 | 40 | 2 | 0 | 0 |
| `burst` | 30 | 30 | 0 | 0 | 0 |
| `sustained_overload` | 26 | 26 | 0 | 0 | 0 |
| `rpm_halved` | 12 | 12 | 0 | 0 | 0 |
| `daily_exhausted` | 12 | 12 | 0 | 0 | 0 |

