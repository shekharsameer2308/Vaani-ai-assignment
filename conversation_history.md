# Conversation History with Assistant: The Relay

> **Session Overview & Working Record:**
> This document provides the engineering conversation history for the development, benchmarking, audit, and submission of **The Relay**. 
> - **Sessions 1–3 (System Design, Implementation & Load Experiments):** Reconstructed from working notes and chronological git commit history (`dffbf217`, `eebe2527`, `02ef83a`).
> - **Session 4 (Audit, Schema Reconciliation, Math Repair & Verification):** Exported from the final working sessions with meta-prompts removed for clarity.

---

## Session 1: Corpus Acquisition, Version Pinning & Extraction

### User Prompt 1.1
```text
I am building a grounded RAG assistant called "The Relay" that answers engineering queries strictly over three specific research papers:
1. "Attention Is All You Need" (arXiv:1706.03762v7)
2. "Moshi: a speech-text foundation model for real-time dialogue" (arXiv:2410.00037v1)
3. "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling" (arXiv:2507.07955v2)

Write an automated download script in Python that fetches these exact PDF versions from arXiv, validates them against hardcoded SHA-256 hashes in papers/manifest.json, and extracts the text page-by-page into corpus/pages.jsonl so we can produce exact page-level citations.
```

### Assistant Response 1.1
- Created [`papers/manifest.json`](papers/manifest.json) containing SHA-256 checksums and exact arXiv download URLs.
- Implemented [`scripts/fetch_papers.py`](scripts/fetch_papers.py) with stream hashing, retries, and corrupted download detection.
- Built [`corpus/extract.py`](corpus/extract.py) extracting 120 page records into [`corpus/pages.jsonl`](corpus/pages.jsonl), preserving document ID, page numbers, and clean text tokens.

---

## Session 2: Pipeline Architecture, Rate Limits & Grounding Gate

### User Prompt 2.1
```text
Now design the core pipeline in Python. We must operate strictly within Groq's free tier limits (30 RPM, 14,400 TPM, 500,000 TPD, and ~5 concurrent requests). 

Key architectural requirements:
1. A decoupled 4-step pipeline: Admit -> Retrieve -> Generate -> Verify -> Respond.
2. Token bucket rate limiter tracking RPM, TPM, and daily quotas in pipeline/limits.py. If saturated, degrade gracefully to extractive snippets instead of crashing.
3. Crucial grounding constraint: DO NOT let the LLM guess whether the text answers the question. Implement an explicit absence threshold gate in the retrieval stage. If BM25 relevance or term coverage falls below the threshold, return "The papers do not say." immediately without calling the LLM.
4. Verify step: Check that all cited tags ([P1]..[Pk]) exist in the retrieved set. If missing, run a single targeted repair loop.
```

### Assistant Response 2.1
- Implemented [`pipeline/limits.py`](pipeline/limits.py) with dual sliding token buckets (requests, tokens) and an asyncio semaphore for concurrency.
- Built [`retrieval/search.py`](retrieval/search.py) implementing Okapi BM25 ranking and absence gating (`absence_threshold = 12.0`).
- Implemented [`pipeline/pipeline.py`](pipeline/pipeline.py) and [`pipeline/verify.py`](pipeline/verify.py) enforcing explicit citation tagging and 1-shot repair turns.

### Human Oversight & Bug Fix (NumPy Array Ambiguity)
- **Error caught by user:** During BM25 scoring in `retrieval/search.py`, `bm25.get_scores(query_tokens)` returned a NumPy `ndarray`. The code evaluated `if not scores: return []`, triggering a Python `ValueError: The truth value of an array with more than one element is ambiguous`.
- **Resolution:** Replaced the boolean truth check with explicit length verification (`if len(scores) == 0:` or `scores.size == 0`) and safe scalar conversion.

---

## Session 3: Calibrated Stand-in Service & Load Experiments

### User Prompt 3.1
```text
We need to test the pipeline under realistic load without exhausting our real Groq free-tier quota. 
1. Build a stand-in LLM service (standin/simulator.py and standin/app.py) calibrated against real Groq measurements. It must NOT use fixed sleeps. Model a lognormal TTFT distribution (p50 ~200ms, p95 >1000ms), stream tokens at 240 tok/s, enforce 30 RPM / 14,400 TPM sliding windows, and add a +15% latency penalty per concurrent in-flight request.
2. Build an open-loop load testing driver (load/run.py) to run 5 scenarios: steady, burst, sustained_overload, rpm_halved, and daily_exhausted.
3. Run an ablation sweep across 3 retrieval settings (Setting A: 250 words top-3; Setting B: 500 words top-5; Setting C: 500 words top-10 rerank) over 15 benchmark questions.
```

### Assistant Response 3.1
- Built [`standin/simulator.py`](standin/simulator.py) modeling the lognormal latency distribution and load equation:
  $\text{TTFT}_{\text{effective}} = \text{TTFT}_{\text{base}} \times (1 + \alpha \cdot \max(0, \text{in-flight} - \text{knee})^\beta)$.
- Implemented [`load/run.py`](load/run.py) using an asynchronous queue worker to record per-request metrics (`results.csv`) and per-stage timings (`results/timings.csv`).
- Executed the 5 load scenarios and the 15-question retrieval ablation sweep.

### Human Overrule (Decoupling Offline Sweep from Limiter)
- **Decision overruled:** The assistant initially attempted to run the offline retrieval quality parameter sweep through the same throttled stand-in client instance used for load testing. This caused quality evaluation queries to be dropped by the 30 RPM limiter.
- **User intervention:** Overruled this design to ensure offline accuracy metrics remained decoupled from synthetic rate-limiting noise; gave `eval/grade.py` its own clean client instance with unconstrained limits.

---

## Session 4: Audit, Schema Reconciliation & Submission Preparation

### User Prompt 4.1 (Project Brief & Cross-Check)
```text
Give a brief of the project and cross-check the parameters against the assignment brief to tell me if it's ready for submission.
```

### Assistant Response 4.1
- Summarized the architecture, paper corpus, stand-in calibration, and evaluation results.
- Identified critical discrepancies:
  1. `results.csv` on disk had been overwritten by a partial run with missing retrieval sweep rows and non-conforming enum names (`refused_incorrectly`, `hallucinated`).
  2. The technical write-up metrics in `SUBMISSION.md` were derived from the earlier 165-row golden dataset.

### User Prompt 4.2 (Correction & Schema Alignment)
```text
Make corrections where required.
```

### Assistant Response 4.2
- Restored the verified 165-row golden dataset to `results.csv` and `results/requests.csv`.
- Confirmed required schema: `source` strictly in `[real, stand-in]`, `answer_grade` strictly in `[correct, wrong, refused_correctly, refused_wrongly]`.
- Updated [`analysis/tables.py`](analysis/tables.py) to ensure all tables (Table 2a, 2b, 3) are recomputable directly from `results.csv` with exact matching percentiles.
- Corrected the $p_{99}$ tail latency request reference in `SUBMISSION.md` Section 4.

### User Prompt 4.3 (KaTeX Math Rendering Fix)
```text
Fix this:
[Image showing KaTeX parse error: '_' allowed only in math mode for \text{in_flight}]
```

### Assistant Response 4.3
- Diagnosed the Markdown unescaping issue where `\text{in\_flight}` was stripped to `\text{in_flight}` before reaching KaTeX.
- Replaced `\text{in_flight}` with `\text{in-flight}` in both [`README.md`](README.md) and [`standin/README.md`](standin/README.md), resolving the render crash.

### User Prompt 4.4 (Mermaid Architecture Diagram)
```text
Correct the mermaid diagram in README.
```

### Assistant Response 4.4
- Replaced the raw ASCII text block in [`README.md`](README.md) with a fully compliant, natively rendered GitHub Mermaid flowchart featuring explicit decision branch labels and quoted node titles.

### User Prompt 4.5 (Final Verification & 2-Page PDF Build)
```text
Check the pre-flight requirements:
1. Verify results.csv column order and table recomputability.
2. Confirm zero API keys appear in git history.
3. Build the final 2-page PDF layout and verify page count.
```

### Assistant Response 4.5
- Verified `git log -p --all` confirmed 0 live API keys committed.
- Validated all 12 columns in `results.csv` against required order.
- Created [`submission.tex`](submission.tex) and built [`scripts/generate_pdf.py`](scripts/generate_pdf.py), generating [`submission.pdf`](submission.pdf) verified at **exactly 2 pages** via `pypdf`.
