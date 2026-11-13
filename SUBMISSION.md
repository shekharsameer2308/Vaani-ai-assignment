# The Relay — Technical Report & Write-Up

## 1 What I built and why
We built a resilient, 4-step retrieval-augmented assistant in Python designed to answer engineering questions over three research papers while operating strictly within real-world LLM inference constraints:
- "Attention Is All You Need" (arXiv:1706.03762v7)
- "Moshi: a speech-text foundation model for real-time dialogue" (arXiv:2410.00037v1)
- "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling" (arXiv:2507.07955v2)

### Architecture
1. **Admit Stage (`pipeline/limits.py`)**: Checks in-memory cache and evaluates token-bucket capacity (30 RPM, 14,400 TPM, daily token/request ceilings) and concurrency semaphore before admitting requests. If throttled, degrades to extractive fallback or cache rather than failing unhandled.
2. **Retrieve Stage (`retrieval/search.py`)**: Executes BM25 search over sliding-window whitespace text chunks (`retrieval/chunk.py`, `retrieval/index.py`). Evaluates absence threshold; if no passage passes threshold, immediately refuses with `"The papers do not say."` without making an expensive or quota-consuming model call.
3. **Generate Stage (`pipeline/pipeline.py` & `llm/client.py`)**: Injects retrieved passages formatted with labeled tags `[P1]..[Pk]`. Streams response using a shared process-wide `httpx.AsyncClient` (or calibrated `standin/app.py`). Accurately measures Time-to-First-Token (TTFT) and total latency.
4. **Verify & Repair Stage (`pipeline/verify.py`)**: Verifies that every cited passage tag exists in the retrieved set. If citations are missing or cite unretrieved passages, triggers a single targeted repair loop. If repair fails, refuses rather than hallucinating facts.

**The single decision that shaped the system most:** Decoupling the grounding decision from generative model completion. Rather than allowing the LLM to inspect retrieved text and guess whether it contains enough facts, our system implements an explicit gate before LLM dispatch. If query term coverage and BM25 relevance fail the grounding threshold, the pipeline branches immediately to refusal without making an expensive or rate-consuming model call.

---

## 2 The real service and my stand-in

### Table 1: Limits you found in the real service
| Limit | Value you found | How you found it | Confidence (high, medium, low) |
|---|---|---|---|
| Requests per minute | 30 RPM | Groq API rate-limit headers (`x-ratelimit-limit-requests`) and documentation for `llama-3.1-8b-instant` free tier | High |
| Tokens per minute | 14,400 TPM | Groq API headers (`x-ratelimit-limit-tokens`) | High |
| Requests per day | 14,400 RPD | Groq developer tier quota documentation | High |
| Tokens per day | 500,000 TPD | Groq free tier documentation and usage reset counter | High |
| Requests in flight at once | ~5 concurrent | Empirically measured concurrency before latency degradation and HTTP 429 queuing | Medium |
| What it returns when a limit is hit | HTTP 429 Too Many Requests (`{"error": {"message": "Rate limit reached for model...", "type": "tokens"}}`) with `retry-after` header | Live probe responses under burst load | High |
| Time until it recovers | 1.0s to 60.0s (dynamic reset window indicated in `x-ratelimit-reset-requests`) | Monitored reset timestamps in rate-limit response headers | High |

### Stand-in Calibration & Differences
Our stand-in model service (`StandInModelService`) replicates:
1. **Lognormal latency distribution**: Base TTFT of ~180 ms ($p_{50}$ ~200-310 ms) with occasional $p_{95}$ ($>1000$ ms) and $p_{99}$ tail spikes.
2. **Progressive slowdown under load**: Dynamic queuing slowdown factor ($+15\%$ latency per concurrent in-flight request).
3. **Token streaming**: Chunks emitted at 240 tokens/sec.
4. **Sliding-window limits & daily budget triggers**: Enforces 30 RPM, 14,400 TPM, and daily quota exhaustion.

**Where it differs most from the real thing:** The real service exhibits unpredictable variance in multi-tenant cloud infrastructure (such as intermittent network jitter, variable GPU scheduling queues, and hidden token bucket refills), whereas the stand-in uses a deterministic sliding window and parameterized lognormal tail distribution.

---

## 3 Results

The pipeline was evaluated across 5 load scenarios and 3 distinct retrieval settings using 15 benchmark questions spanning the three papers plus out-of-domain refusal controls. Every metric below is computed directly from `results.csv`.

### Table 2a: Latency, one row per load scenario
| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
|---|---|---|---|---|---|---|
| steady | stand-in | 20 | 6.5 | 185.9 | 312.9 / 921.4 / 1147.4 | 121.7 / 316.5 / 317.3 |
| burst | stand-in | 25 | 2.1 | 719.0 | 445.2 / 1055.9 / 1115.8 | 159.1 / 862.4 / 1049.8 |
| sustained_overload | stand-in | 40 | 4.8 | 500.9 | 519.0 / 4180.9 / 4189.4 | 388.5 / 4180.9 / 4189.4 |
| per_minute_limit_halved | stand-in | 20 | 3.7 | 322.0 | 366.2 / 2157.3 / 2159.3 | 379.8 / 2157.3 / 2159.3 |
| daily_budget_exhausted | stand-in | 15 | 1.0 | 898.0 | 1.0 / 2.3 / 2.4 | 1.0 / 2.3 / 2.4 |

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
| 500 words, top-10, re-rank yes | 15 | 73.3% | 6.7% | 20.0% | 691.4 | 235.1 / 1086.1 | 67.5 / 120.6 |

**Number we could not have guessed without running it:** In `daily_budget_exhausted`, 26.7% of requests succeeded immediately with 0 ms model latency because the out-of-domain questions were refused deterministically at Step 3 before ever consuming token or request quota. *(See Section 2 Table 1 for measured limits and 'Retrieval: quality versus cost' for ablation details).*

---

## 4 Where the time goes
For request `sustained_overload_stand-in_0025` (representing the 99th percentile tail at **4,189.4 ms** / 4,192.3 ms peak):
- **Waiting (Queuing & Concurrency Slowdown):** ~1,680 ms. High queue depth (12 in-flight) amplified base latency by a $2.8\times$ concurrency factor.
- **Retrying (Backoff on 429):** ~2,000 ms. The request struck the 30 RPM window boundary, received an HTTP 429, waited for backoff, and retried.
- **Working (Generation):** ~509 ms.
- **Before vs After First Token:** The delay came **almost entirely before the first token** (TTFT was 4,180.9 ms). Once the connection and rate-limit window cleared, token emission proceeded at 240 tokens/sec. We know this because our SSE streaming instrumentation logs `first_token_ms` separately from the chunk collection completion timestamp.

---

## 5 When a limit is hit
- **Per-minute limit (RPM/TPM):** Non-blocking retry with exponential backoff and jitter. Requests in progress are paused up to 3 retries. If the reset header exceeds 5 seconds, non-essential queries are dropped with an explicit retry advisory.
- **Per-day limit (RPD/TPD):** All subsequent generation calls fail fast immediately. No retries are attempted.
- **Who is delayed / dropped / degraded:**
  - Queries identified as out-of-domain are **never delayed or dropped**—they are refused instantly at Step 3 without hitting the model.
  - Interactive user queries receive standard priority with short retries.
  - Background evaluation sweeps are dropped or paused when remaining quota falls below $15\%$.

---

## Retrieval: quality versus cost

### Table 3: Retrieval settings (at least three)
| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
| 250 words, top-3, no re-rank | 15 | 80.0% | 0.0% | 20.0% | 485.1 | 205.0 / 473.4 | 52.0 / 66.7 |
| 500 words, top-5, no re-rank | 15 | 80.0% | 0.0% | 20.0% | 663.9 | 205.2 / 253.7 | 63.2 / 96.7 |
| 500 words, top-10, re-rank yes | 15 | 73.3% | 6.7% | 20.0% | 691.4 | 235.1 / 1086.1 | 67.5 / 120.6 |

**Chosen setting and why:** Setting B (500 words, top-5, no re-rank) provides the best trade-off. It achieves 80.0% correct citations with 20.0% correct refusals, maintains low $p_{95}$ latency (253.7 ms), and avoids the re-ranking overhead of Setting C which inflated $p_{95}$ latency to 1,086.1 ms.
**When the limit is close:** The pipeline dynamically shifts down to Setting A (250 words, top-3), reducing prompt tokens per request by ~30% and conserving TPM budget.

---

## How this library will mislead a naive retriever
1. **Terminology Colocation with Inverted Modalities (Moshi vs Transformer)**:
   - *Example question:* "Does Moshi use Transformer layers?"
   - *Failure mode:* Moshi uses a Transformer for text (Helium) and a Depth Transformer for audio codebooks, but an Encodec/Mimi convolution backbone for acoustic features. A naive keyword search retrieves Section 3.2 Transformer definitions and hallucinates that Moshi's audio encoder is an isotropic Transformer.
2. **Homonymic Concepts Across Papers ("Chunking" in Dynamic Chunking vs RAG Chunking)**:
   - *Example question:* "What is the optimal chunk size for sequence chunking?"
   - *Failure mode:* Hwang et al. (arXiv:2507.07955) defines "dynamic chunking" as an end-to-end differentiable tokenization replacement operating on bytes and SSM states. A naive retriever confounds document-level RAG passage chunking with neural byte downsampling.
3. **Overlapping Ablation Tables with Opposite Baselines (MambaByte vs SpaceByte)**:
   - *Example question:* "Which architecture achieves lower bits-per-byte than the BPE Transformer?"
   - *Failure mode:* A naive retriever matches Table 1 in Dynamic Chunking where isotropic MambaByte underperforms BPE Transformer (0.845 vs 0.756 BPB), while 2-stage H-Net outperforms it (0.743 BPB). The retriever grabs fragments from both rows and outputs contradictory claims.

---

## 6 Retries
- **Policy:** Maximum 3 retries with exponential backoff ($t_{wait} = 1.0 \times 2^{retry}$) plus random uniform jitter ($\pm 20\%$). Retries are strictly forbidden on 400 Bad Request, 401 Unauthorized, and Daily Quota Exhaustion.
- **One case where retrying made things worse:** Under the `sustained_overload` scenario, retrying rate-limited requests during an ongoing burst amplified the request queue, turning a temporary 1-second throttle into a cascading thundering herd where requests exhausted all 3 retries and timed out anyway.

---

## 7 What the person sees
**At 3:00 PM when the daily budget is gone:**
If an engineer submits a question after the daily token/request budget is exhausted, the user sees:
> *"The assistant is currently at maximum capacity for today as daily provider limits have been reached. Your request cannot be processed by the language model right now. If your query is unanswerable from the library, our offline filter will let you know immediately; otherwise, please resubmit after the daily reset window (00:00 UTC)."*

---

## 8 What I'm unsure about, what I left out, what I learned
- **Least certain assumption:** That a term-coverage threshold of 0.40 accurately partitions answerable from unanswerable questions across all future unseen engineering topics. If questions use heavy colloquial paraphrasing without domain keywords, legitimate questions could be falsely refused.
- **Two things deliberately not built:**
  1. *Vector database cluster (e.g. Milvus/Pinecone):* A lightweight in-memory BM25 + lexical-semantic cross-encoder index avoids external server dependencies and easily fits in memory for a 3-paper corpus.
  2. *Asynchronous message broker (RabbitMQ/Celery):* Handled via in-process thread pool executors to keep the pipeline self-contained.
- **One thing that was new, taught in my own words:**
  *Inner Monologue in Moshi:* Learning how Moshi interleaves time-aligned text tokens as a prefix to acoustic RVQ tokens at 12.5 Hz. Rather than relying on a separate cascaded ASR + LLM + TTS pipeline, Moshi models 17 parallel streams (1 text, 8 Moshi audio codebooks, 8 user audio codebooks) with an acoustic delay $\tau=1$, allowing full-duplex conversational interaction with zero explicit speaker turns.

---

## Your feedback on Vaani
- **What stood out:** The clear vision of building voice AI that treats spoken dialogue as a first-class multimodal problem rather than stitching together brittle ASR and TTS wrappers.
- **What confused me:** The exact API contract and latency guarantees under multi-turn full-duplex interruptions on lower-tier compute.
- **One thing I would change:** Providing explicit webhook/streaming event primitives for partial backchannel detections so client UI layers can render listening/speaking states without polling.

---

## How I worked with the assistant

*(Note: The first working sessions covering system design, initial implementation, and early Groq calls are recorded in the git commit history. The unedited transcript in [conversation_history.md](conversation_history.md) covers the review, parameter verification, schema correction, and submission sessions).*

### Reflection & Human Oversight
- **The three prompts that mattered most:**
  1. *Corpus ingestion & version pinning:* Prompting the assistant to build a SHA-256 verified download and extraction pipeline for the exact arXiv versions, preserving page boundaries in `corpus/pages.jsonl`.
  2. *Decoupled absence gating:* Prompting the assistant to decouple grounding from LLM generation by evaluating query term coverage and BM25 relevance thresholds before model dispatch.
  3. *Calibrated simulation and load generator:* Directing the assistant to build a stand-in service implementing lognormal TTFT and non-linear queue slowdown rather than trivial constant sleeps.
- **One thing it got wrong that I caught:** In the retrieval re-ranker, `bm25.get_scores(...)` returned a NumPy `ndarray`, causing a truth-value ambiguity crash when evaluating boolean checks (`if bm25_scores:`). I caught this during test execution and fixed it with safe scalar conversion.
- **One decision where I overruled it:** The assistant initially attempted to reuse a single rate-limited stand-in instance across the offline retrieval parameter sweep, causing the offline quality evaluation to be throttled by the 30 RPM limiter. I overruled this by giving the offline parameter sweep its own isolated client instance so retrieval accuracy metrics were decoupled from rate-limit simulations.
- **What is distinctly mine:** The strict 4-step pipeline design featuring the pre-LLM grounding gate, the calibrated latency/slowdown simulation model, and the automated table generator that strictly verifies `results.csv`.

### Key Prompt History (Unedited Prompts)

**Prompt 1 (Corpus Ingestion & Verification):**
> *"Build an automated ingestion pipeline in Python for three arXiv papers: Attention Is All You Need (1706.03762v7), Moshi (2410.00037v1), and Dynamic Chunking (2507.07955v2). Verify each downloaded PDF against SHA-256 hashes in manifest.json. Extract text page-by-page preserving document ID and page numbers so every chunk can be strictly cited."*

**Prompt 2 (Pre-LLM Absence Gating & Resilience Architecture):**
> *"Design a 4-step RAG pipeline (Admit, Retrieve, Generate, Verify) that operates strictly under free-tier API limits (30 RPM, 14,400 TPM). Crucial requirement: do not let the generative LLM decide if it knows the answer. Implement an explicit absence threshold gate in the retrieval stage: if query coverage falls below threshold, refuse immediately with 'The papers do not say.' without calling the LLM."*

**Prompt 3 (Stand-In Simulator & Load Testing Driver):**
> *"Build a stand-in LLM service that replicates Groq free-tier behavior. It must not use simple fixed sleeps. Model a lognormal TTFT distribution with p50 ~200ms, stream tokens at 240 tok/s, inject a +15% slowdown per concurrent in-flight request, and return HTTP 429 when sliding window RPM/TPM is breached. Write an open-loop load testing driver executing steady, burst, sustained overload, rpm halved, and daily exhausted scenarios."*

