# The Relay: Resilient Grounded RAG Pipeline

The Relay is a production-grade question-answering pipeline designed to answer engineering queries strictly grounded in three research papers: *Attention Is All You Need* (arXiv:1706.03762v7), *Moshi* (arXiv:2410.00037v1), and *Dynamic Chunking* (arXiv:2507.07955v2). Operating under strict free-tier rate limits (30 RPM, 14,400 TPM, and daily quotas), the system decouples admission and grounding decisions from generative completion, guaranteeing exact passage citations and refusing queries that fall outside the library. The project includes an open-loop load shaping driver and an empirically calibrated stand-in inference service that models lognormal Time-to-First-Token (TTFT), non-linear queuing slowdown, and fault injection.

## Status

- **Paper Ingestion & Corpus Extraction:** Complete. All 3 exact PDF versions are downloaded via [`scripts/fetch_papers.py`](scripts/fetch_papers.py), verified against SHA-256 hashes in [`papers/manifest.json`](papers/manifest.json), and extracted into 120 page records in [`corpus/pages.jsonl`](corpus/pages.jsonl).
- **BM25 Retrieval & Absence Gating:** Complete. Chunking, index caching in `data/index.pkl`, and absence threshold gating are implemented and covered by unit tests in [`tests/test_retrieval.py`](tests/test_retrieval.py).
- **Core Pipeline (Admit, Retrieve, Generate, Verify, Respond):** Complete. Implemented as pure async functions in [`pipeline/pipeline.py`](pipeline/pipeline.py) with structured tracing, token-bucket admission, and citation verification.
- **Stand-in Model Service:** Complete & Calibrated. Simulates lognormal latency, token streaming, rate limits, and load curve in [`standin/simulator.py`](standin/simulator.py) and [`standin/app.py`](standin/app.py).
- **Rate-Limit & Stress Testing:** Measured. 5 traffic scenarios (`steady`, `burst`, `sustained_overload`, `rpm_halved`, `daily_exhausted`) have been executed and recorded into [`results/requests.csv`](results/requests.csv) and [`results/timings.csv`](results/timings.csv).
- **Retrieval Parameter Sweeps:** Measured. Evaluated across 3 distinct settings on 15 benchmark questions; blind grading sheet generated in [`eval/grading_sheet.csv`](eval/grading_sheet.csv).
- **Real Groq Provider Probing:** Probe script implemented in [`llm/probe_groq.py`](llm/probe_groq.py); live probing requires an opt-in `GROQ_API_KEY`.

## Quick Start

Follow these steps to set up the environment, verify the papers, build the retrieval index, and execute the test suite:

```bash
# 1. Clone the repository
git clone https://github.com/shekharsameer2308/Vaani-ai-assignment..git
cd "Vaani-ai-assignment."

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (optional for real Groq API; not required for tests or standin)
export GROQ_API_KEY=""

# 5. Fetch and verify paper PDFs
python scripts/fetch_papers.py

# 6. Extract corpus pages to JSONL
python corpus/extract.py

# 7. Build and cache the BM25 index
python -m retrieval.index

# 8. Run the test suite
python -m unittest discover tests
```

To run an interactive CLI query:
```bash
python -m retrieval.search "What is the formula for scaled dot product attention?" --piece 200 --k 3
```

## Architecture

```text
               +-------------------------------------------+
               | User Question                             |
               +-------------------------------------------+
                                     |
                                     v
                        [ 1. Admit Stage ]
                       pipeline/limits.py
                   - In-memory cache hit?
                   - Token buckets (RPM, TPM)
                   - In-flight semaphore
                   - Daily budget exhausted?
                        /                         \
           (Throttled / Exhausted)           (Admitted)
                      /                             \
                     v                               v
         [ Extractive / Daily Fallback ]    [ 2. Retrieve Stage ]
               pipeline/fallback.py          retrieval/search.py
                                            - BM25 Okapi top-k
                                            - Absence threshold check
                                                    /         \
                                        (No passages)     (Passages found)
                                                /                 \
                                               v                   v
                                      [ Explicit Refusal ]  [ 3. Generate Stage ]
                                     "The papers do not say"  pipeline/pipeline.py
                                                              - Shared httpx client
                                                              - [P1]..[Pk] prompt tags
                                                              - TTFT & usage logging
                                                              - llm/retry.py on error
                                                                       |
                                                                       v
                                                            [ 4. Verify & Repair ]
                                                              pipeline/verify.py
                                                              - Verify [Pi] citations
                                                              - Targeted repair prompt
                                                                       |
                                                                       v
                                                            [ 5. Respond Stage ]
                                                              pipeline/pipeline.py
                                                              - Formatted response
                                                              - Cache answer
                                                              - Execution trace
```

- **Admit Stage ([`pipeline/limits.py`](pipeline/limits.py)):** Checks semantic cache and evaluates token-bucket capacity (30 RPM, 14,400 TPM, daily token/request ceilings) and concurrency semaphore before admitting requests. Throttled requests gracefully degrade to extractive fallbacks or cached answers rather than failing unhandled.
- **Retrieve Stage ([`retrieval/search.py`](retrieval/search.py)):** Performs BM25 ranking over sliding-window chunks ([`retrieval/chunk.py`](retrieval/chunk.py), [`retrieval/index.py`](retrieval/index.py)). Evaluates an absence threshold; if no passage satisfies the relevance threshold, the pipeline refuses immediately with `"The papers do not say."` without consuming provider tokens.
- **Generate Stage ([`pipeline/pipeline.py`](pipeline/pipeline.py)):** Formats retrieved passages with explicit `[P1]..[Pk]` tags. Calls the LLM via [`llm/client.py`](llm/client.py) or [`standin/app.py`](standin/app.py). On transient connection errors or 429 rate limits, applies pure retry decisions from [`llm/retry.py`](llm/retry.py).
- **Verify Stage ([`pipeline/verify.py`](pipeline/verify.py)):** Inspects generated text for passage citations. If citations are omitted or cite unretrieved tags, triggers a single targeted repair turn. If repair fails, refuses rather than hallucinating.
- **Respond Stage ([`pipeline/pipeline.py`](pipeline/pipeline.py)):** Assembles the structured `Result` containing the grounded answer, verified citations, status, token counts, and step-by-step latency trace.
- **Rate Control and Degradation Modes:** Rate limits are tracked via token buckets with non-blocking acquisition timeouts. When saturated, queries degrade gracefully via [`pipeline/fallback.py`](pipeline/fallback.py) by quoting top retrieved passages with a clear status disclaimer.

## Reproducing the Results

To reproduce all load shapes and evaluation metrics:

```bash
# 1. Run all 5 traffic scenarios (steady, burst, sustained_overload, rpm_halved, daily_exhausted)
python -m load.run --scenario all --source standin --retrieval-setting setting_a

# 2. Run retrieval ablation parameter sweep across the 15 benchmark questions
python -m eval.grade --setting all

# 3. Generate analysis Markdown tables
python -m analysis.tables --csv results.csv --output results/tables.md
```

- `load/run.py` writes individual request metrics to [`results/requests.csv`](results/requests.csv) (and mirrors to `results.csv`), timing traces to [`results/timings.csv`](results/timings.csv), and experiment metadata to [`results/run.json`](results/run.json).
- `eval/grade.py` outputs evaluation metrics to [`results/eval_results.json`](results/eval_results.json) and blind review items to [`eval/grading_sheet.csv`](eval/grading_sheet.csv).
- `analysis/tables.py` reads the CSV results and compiles Tables 1, 2a, 2b, and 3 into [`results/tables.md`](results/tables.md).

## Results

Full performance and grounding analysis tables are located in [`results/tables.md`](results/tables.md) and experiment metadata in [`results/run.json`](results/run.json).

> *Every number in tables.md is computed from the CSVs in results/ by analysis/tables.py.*

Key measured highlights from [`results/tables.md`](results/tables.md):
- **Load Shapes (Table 1):** Under `steady` load, median TTFT was 15.5 ms with 0.0% unhandled errors. Under `burst` load (3x limit), the system maintained 0.0% unhandled errors by admitting requests through the backpressure queue and absorbing throttled requests via fallback modes.
- **Retrieval Ablation (Table 2a & 2b):** Setting A (`piece=200, k=3`) achieved 93.3% precision on the 15 benchmark questions while using an average of only 125 prompt tokens per request, compared to 1,129 tokens for Setting B (`piece=500, k=5`) and 1,175 tokens for Setting C (`piece=500, k=10`).
- **Resilience (Table 3):** Under `sustained_overload` and `rpm_halved`, zero unhandled exceptions were emitted; requests were cleanly processed either via direct completion or graceful fallback.

## Limitations and Differences from Real Service

A comprehensive analysis of how the mock inference service differs from the real Groq API is documented in [`standin/README.md`](standin/README.md).

Summary of differences:
1. **Network Jitter:** The stand-in runs in-process with zero network overhead, whereas real Groq calls traverse TLS and edge proxies.
2. **Deterministic Heuristics:** The stand-in emits deterministic grounded answers and negative refusals to enable reproducible unit test assertions.
3. **Queue Model:** The stand-in models queuing via an empirical load equation: $\text{TTFT} = \text{base} \times (1 + \alpha \cdot \max(0, \text{in\_flight} - \text{knee})^\beta)$, whereas real datacenter latency spikes depend on multi-tenant hardware contention.

## Repository Layout

| Directory / File | Description |
| :--- | :--- |
| `analysis/` | Table generation script [`analysis/tables.py`](analysis/tables.py) producing [`results/tables.md`](results/tables.md). |
| `configs/` | System limits ([`configs/limits.yaml`](configs/limits.yaml)) and pricing models ([`configs/pricing.yaml`](configs/pricing.yaml)). |
| `corpus/` | Corpus extraction script [`corpus/extract.py`](corpus/extract.py) and extracted data [`corpus/pages.jsonl`](corpus/pages.jsonl). |
| `data/` | Built BM25 parameter sidecar [`data/index_meta.json`](data/index_meta.json); index pickle is rebuilt dynamically. |
| `eval/` | Benchmark suite: 15 questions ([`eval/questions.json`](eval/questions.json)), grader ([`eval/grade.py`](eval/grade.py)), blind review sheet ([`eval/grading_sheet.csv`](eval/grading_sheet.csv)). |
| `llm/` | Shared HTTP client ([`llm/client.py`](llm/client.py)), retry policy ([`llm/retry.py`](llm/retry.py)), and endpoint probe ([`llm/probe_groq.py`](llm/probe_groq.py)). |
| `load/` | Load driver ([`load/run.py`](load/run.py)) and traffic scenarios configuration ([`load/scenarios.yaml`](load/scenarios.yaml)). |
| `papers/` | Paper manifest ([`papers/manifest.json`](papers/manifest.json)) tracking PDF hashes and page counts. |
| `pipeline/` | Core 4-step pipeline ([`pipeline/pipeline.py`](pipeline/pipeline.py)), limits ([`pipeline/limits.py`](pipeline/limits.py)), verification ([`pipeline/verify.py`](pipeline/verify.py)), and fallbacks ([`pipeline/fallback.py`](pipeline/fallback.py)). |
| `results/` | Output directory containing `requests.csv`, `timings.csv`, `run.json`, `eval_results.json`, and `tables.md`. |
| `retrieval/` | Whitespace token chunker ([`retrieval/chunk.py`](retrieval/chunk.py)), BM25 builder ([`retrieval/index.py`](retrieval/index.py)), and search CLI ([`retrieval/search.py`](retrieval/search.py)). |
| `scripts/` | Paper download and verification script [`scripts/fetch_papers.py`](scripts/fetch_papers.py). |
| `standin/` | Calibrated stand-in inference service ([`standin/simulator.py`](standin/simulator.py), [`standin/app.py`](standin/app.py), [`standin/config.yaml`](standin/config.yaml), [`standin/README.md`](standin/README.md)). |
| `tests/` | Unit test suites for retrieval ([`tests/test_retrieval.py`](tests/test_retrieval.py)) and retry policies ([`tests/test_retry.py`](tests/test_retry.py)). |
| `SUBMISSION.md` | Formal 2-page engineering technical write-up. |

## Conversation History

The complete interaction trajectory and development audit logs are archived under the assistant conversation directory:
- App Data Path: `~/.gemini/antigravity/brain/c7c17117-456f-43bb-89fa-c102bab05600/.system_generated/logs/transcript.jsonl`
- Conversation ID: `c7c17117-456f-43bb-89fa-c102bab05600`

## Notes on the Write-Up

The dense 2-page technical write-up required by the brief is located at [`SUBMISSION.md`](SUBMISSION.md). It addresses:
1. What I built and why
2. The real service and my stand-in (with limit measurements)
3. Results (load scenarios and outcome breakdowns)
4. Where the time goes (TTFT vs generation latency breakdown)
5. When a limit is hit (degradation and priority policies)
6. Retrieval: quality versus cost & naive retriever pitfalls
7. Retries policy & failure cases
8. What the person sees during quota exhaustion
9. Unsure assumptions, omissions, and conceptual takeaways (e.g., Moshi Inner Monologue)
10. Feedback on Vaani and collaboration methodology

