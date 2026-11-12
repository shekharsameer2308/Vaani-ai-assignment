# Stand-in Model Service

This directory provides the calibrated mock/stand-in model service simulating Groq API inference dynamics for load testing, scenario experiments, and CI/CD pipelines without consuming quota.

## Architectural Responsibilities

1. `standin/config.yaml`: Defines simulation parameters including lognormal TTFT fit ($\mu, \sigma$), token streaming rate, sliding-window rate limit ceilings, load curve parameters, fault probabilities, and deterministic seed.
2. `standin/simulator.py`: Implements queuing math, token buckets, lognormal sampling, and progressive slowdown under concurrency:
   $$\text{TTFT}_{\text{effective}} = \text{TTFT}_{\text{base}} \times \left(1 + \alpha \cdot \max(0, \text{in-flight} - \text{knee})^\beta\right)$$
3. `standin/app.py`: Client interface (`StandInLLMClient`) offering identical signature to `GroqAsyncClient.stream_chat(...)`, returning `StreamResponse` with exact token counts, statuses, and timing headers.

## Differences Between Stand-in Service and Real Groq API

| Characteristic | Real Groq API (`llama-3.1-8b-instant`) | Stand-in Service (`StandInLLMClient`) |
| :--- | :--- | :--- |
| **Execution Environment** | Hosted LPUs at `api.groq.com/openai/v1` over TLS network transport. | In-process asynchronous coroutine; zero external network dependencies. |
| **TTFT Generation** | Dependent on real-time hardware queueing, datacenter load, and networking jitter. | Sampled from a fitted lognormal distribution ($\mu=5.25, \sigma=0.35$) scaled by concurrency load curve. |
| **Token Streaming Rate** | Real hardware token emission (~200–280 tokens/sec depending on system load). | Clocked asynchronously at `tokens_per_second` (default 240 tok/s), scaled by `time_dilation`. |
| **Output Text Generation** | Full LLM transformer generation, capable of stylistic variations and subtle hallucination. | Grounded extractive heuristic: constructs citations to `[P1]` if question matches in-domain keywords; emits `"The papers do not say."` for negative controls. |
| **Quota Resets** | Groq sliding-window rate limit buckets reset via rolling server clocks; daily caps reset at 00:00 UTC. | Sliding window maintained in local double-ended timestamp queues (`request_window`, `token_window`). Explicit daily quota reset triggers. |
| **Failure Modes** | Network timeouts, Cloudflare 502/503 errors, HTTP 429 `rate_limit_exceeded`. | Configurable probabilistic fault injection (`p_500`, `p_drop`, `p_slowdown`, `p_rate_limit`). |
| **Simulation Speedup** | Cannot be accelerated; runs strictly in wall-clock time. | Supports `time_dilation < 1.0` to run accelerated synthetic load tests safely within seconds. |
