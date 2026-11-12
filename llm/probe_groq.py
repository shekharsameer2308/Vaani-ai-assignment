"""Probe Groq API to discover actual models, limits, and latency distribution."""

import argparse
import asyncio
import csv
from datetime import datetime
import json
import os
import sys
import time

from llm.client import GroqAsyncClient

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "probe")


async def run_probe(find_limit: bool = False, accept_quota: bool = False, chosen_model: str = "llama-3.1-8b-instant"):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable is not set.")
        print("Set it via: export GROQ_API_KEY='<your_api_key>'")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    client = GroqAsyncClient(api_key=api_key)

    try:
        # Part 1: GET /models
        print("\n--- Part 1: Querying GET /models ---")
        status, models_data, headers = await client.list_models()
        if status != 200:
            print(f"Failed to fetch models: HTTP {status} {models_data}")
            sys.exit(1)

        models_path = os.path.join(RESULTS_DIR, "models.json")
        with open(models_path, "w", encoding="utf-8") as f:
            json.dump(models_data, f, indent=2)
        print(f"Saved {len(models_data.get('data', []))} models to {models_path}")

        # Filter chat models
        chat_models = [m["id"] for m in models_data.get("data", []) if "whisper" not in m["id"] and "guard" not in m["id"]]
        print(f"Available chat models: {chat_models}")
        print(f"Proposed model: {chosen_model}")

        # Part 2: One streaming call
        print(f"\n--- Part 2: Single calibration call with '{chosen_model}' ---")
        resp = await client.stream_chat(
            messages=[{"role": "user", "content": "Reply with OK"}],
            model=chosen_model,
        )
        print(f"Status: {resp.status} (HTTP {resp.status_code})")
        print(f"Content: {repr(resp.content)}")
        print(f"Time to headers: {resp.time_to_headers_ms} ms")
        print(f"First token (TTFT): {resp.first_token_ms} ms")
        print(f"Total time: {resp.total_ms} ms")
        print(f"Usage block: prompt_tokens={resp.usage.prompt_tokens}, completion_tokens={resp.usage.completion_tokens}, total={resp.usage.total_tokens}")
        print("Rate-limit headers captured:")
        for k, v in sorted(resp.rate_limit_headers.items()):
            print(f"  {k}: {v}")

        # Part 3: N=20 sequential calls, 1s apart
        print(f"\n--- Part 3: N=20 sequential calls, 1s apart ---")
        raw_csv_path = os.path.join(RESULTS_DIR, "raw.csv")
        csv_headers = [
            "request_id",
            "send_time",
            "time_to_headers_ms",
            "first_token_ms",
            "total_ms",
            "status",
            "tokens_in",
            "tokens_out",
            "rate_limit_remaining_requests",
            "rate_limit_remaining_tokens",
            "rate_limit_reset_requests",
            "rate_limit_reset_tokens",
        ]

        rows = []
        for i in range(1, 21):
            req_id = f"probe_seq_{i:02d}"
            send_t = datetime.utcnow().isoformat() + "Z"
            r = await client.stream_chat(
                messages=[{"role": "user", "content": "Reply with OK"}],
                model=chosen_model,
            )
            rem_req = r.rate_limit_headers.get("x-ratelimit-remaining-requests", "")
            rem_tok = r.rate_limit_headers.get("x-ratelimit-remaining-tokens", "")
            reset_req = r.rate_limit_headers.get("x-ratelimit-reset-requests", "")
            reset_tok = r.rate_limit_headers.get("x-ratelimit-reset-tokens", "")

            row = [
                req_id,
                send_t,
                r.time_to_headers_ms,
                r.first_token_ms,
                r.total_ms,
                r.status,
                r.usage.prompt_tokens,
                r.usage.completion_tokens,
                rem_req,
                rem_tok,
                reset_req,
                reset_tok,
            ]
            rows.append(row)
            print(f"  [{i}/20] TTFT: {r.first_token_ms}ms, Total: {r.total_ms}ms, RemReq: {rem_req}, RemTok: {rem_tok}")
            if i < 20:
                await asyncio.sleep(1.0)

        with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
            writer.writerows(rows)
        print(f"Wrote {len(rows)} records to {raw_csv_path}")

        # Part 4: Find minute limit (opt-in)
        if find_limit:
            if not accept_quota:
                print("\n[!] --find-minute-limit requires --i-accept-quota-use to execute.")
                print("    Expected request count: up to 35 back-to-back requests.")
                return

            print("\n--- Part 4: Measuring per-minute limit (back-to-back until HTTP 429) ---")
            count = 0
            while True:
                count += 1
                r = await client.stream_chat(
                    messages=[{"role": "user", "content": "Reply with OK"}],
                    model=chosen_model,
                )
                print(f"  Request #{count}: HTTP {r.status_code} ({r.status})")
                if r.status_code == 429:
                    print("\n[!] Reached rate limit!")
                    print(f"  Count sent before limit: {count}")
                    print(f"  Retry-After: {r.retry_after_s}")
                    print(f"  Rate-limit headers: {r.rate_limit_headers}")
                    print(f"  Error body verbatim:\n{r.error_body}")
                    limit_path = os.path.join(RESULTS_DIR, "rate_limit_429.json")
                    with open(limit_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "requests_sent": count,
                            "retry_after": r.retry_after_s,
                            "headers": r.rate_limit_headers,
                            "error_body": r.error_body,
                        }, f, indent=2)
                    print(f"  Saved 429 trace to {limit_path}")
                    break
                if count > 45:
                    print("Reached 45 requests without 429. Halting.")
                    break

    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(description="Groq API Probe & Limit Discovery")
    parser.add_argument("--find-minute-limit", action="store_true", help="Fire back-to-back to find exact 429 threshold")
    parser.add_argument("--i-accept-quota-use", action="store_true", help="Explicit consent flag for quota use")
    parser.add_argument("--model", type=str, default="llama-3.1-8b-instant", help="Model to probe")

    args = parser.parse_args()
    asyncio.run(run_probe(
        find_limit=args.find_minute_limit,
        accept_quota=args.i_accept_quota_use,
        chosen_model=args.model,
    ))


if __name__ == "__main__":
    main()
