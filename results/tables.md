# Performance & Grounding Analysis Tables

### Table 2a: Latency, one row per load scenario

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
|---|---|---|---|---|---|---|
| steady | stand-in | 20 | 6.5 | 184.6 | 312.9 / 921.4 / 1147.4 | 121.7 / 316.5 / 317.3 |
| burst | stand-in | 25 | 2.1 | 714.3 | 445.2 / 1055.9 / 1115.8 | 159.1 / 862.4 / 1049.8 |
| sustained_overload | stand-in | 40 | 4.8 | 500.0 | 519.0 / 4180.9 / 4189.4 | 388.5 / 4180.9 / 4189.4 |
| per_minute_limit_halved | stand-in | 20 | 3.7 | 324.3 | 366.2 / 2157.3 / 2159.3 | 379.8 / 2157.3 / 2159.3 |
| daily_budget_exhausted | stand-in | 15 | 1.0 | 900.0 | 1.0 / 2.3 / 2.4 | 1.0 / 2.3 / 2.4 |

### Table 2b: Outcomes, same rows

| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|
| steady | 55.0% | 45.0% | 0.0% | 1.40 | 1 | 480.2 | $0.00 (Free Tier) |
| burst | 84.0% | 16.0% | 0.0% | 0.52 | 10 | 589.7 | $0.00 (Free Tier) |
| sustained_overload | 62.5% | 37.5% | 0.0% | 1.12 | 12 | 498.0 | $0.00 (Free Tier) |
| per_minute_limit_halved | 60.0% | 40.0% | 0.0% | 1.20 | 6 | 484.9 | $0.00 (Free Tier) |
| daily_budget_exhausted | 26.7% | 73.3% | 0.0% | 0.00 | 1 | 17.0 | $0.00 (Free Tier) |

### Table 3: Retrieval settings (at least three)

| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
| 250 words, top-3, no re-rank | 15 | 80.0% | 0.0% | 20.0% | 485.1 | 205.0 / 473.4 | 52.0 / 66.7 |
| 500 words, top-5, no re-rank | 15 | 80.0% | 0.0% | 20.0% | 663.9 | 205.2 / 253.7 | 63.2 / 96.7 |
| 500 words, top-10, re-rank yes | 15 | 73.3% | 0.0% | 20.0% | 691.4 | 235.1 / 1086.1 | 67.5 / 120.6 |

