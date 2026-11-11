import csv
from datetime import datetime
import numpy as np
import os
from typing import Dict, List, Any, Tuple


def compute_percentiles(values: List[float], percentiles=(50, 95, 99)) -> Dict[int, float]:
    if not values:
        return {p: 0.0 for p in percentiles}
    arr = np.array(values)
    return {p: float(np.percentile(arr, p)) for p in percentiles}


def parse_results_csv(csv_path: str = "results.csv") -> List[Dict[str, Any]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "request_id": r["request_id"],
                "scenario": r["scenario"],
                "source": r["source"],
                "start_time": r["start_time"],
                "first_token_ms": float(r["first_token_ms"]),
                "total_ms": float(r["total_ms"]),
                "status": r["status"],
                "retries": int(r["retries"]),
                "tokens_in": int(r["tokens_in"]),
                "tokens_out": int(r["tokens_out"]),
                "retrieval_setting": r["retrieval_setting"],
                "answer_grade": r["answer_grade"],
            })
    return rows


def generate_table2(rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    scenarios = ["steady", "burst", "sustained_overload", "per_minute_limit_halved", "daily_budget_exhausted"]
    
    t2a_lines = [
        "### Table 2a: Latency, one row per load scenario",
        "| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |",
        "|---|---|---|---|---|---|---|",
    ]

    t2b_lines = [
        "### Table 2b: Outcomes, same rows",
        "| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for sc in scenarios:
        sc_rows = [r for r in rows if r["scenario"] == sc]
        if not sc_rows:
            continue

        n_req = len(sc_rows)
        source = sc_rows[0]["source"]

        # Duration
        times = [datetime.fromisoformat(r["start_time"].replace("Z", "+00:00")) for r in sc_rows]
        min_t, max_t = min(times), max(times)
        duration_s = max(1.0, (max_t - min_t).total_seconds()) + (sc_rows[-1]["total_ms"] / 1000.0)
        offered_rate = round((n_req / duration_s) * 60.0, 1)

        # Latencies
        ttft_vals = [r["first_token_ms"] for r in sc_rows if r["status"] == "ok" or r["first_token_ms"] > 0]
        ttft_p = compute_percentiles(ttft_vals)
        total_vals = [r["total_ms"] for r in sc_rows]
        total_p = compute_percentiles(total_vals)

        ttft_str = f"{ttft_p[50]:.1f} / {ttft_p[95]:.1f} / {ttft_p[99]:.1f}"
        total_str = f"{total_p[50]:.1f} / {total_p[95]:.1f} / {total_p[99]:.1f}"

        t2a_lines.append(
            f"| {sc} | {source} | {n_req} | {duration_s:.1f} | {offered_rate} | {ttft_str} | {total_str} |"
        )

        # Outcomes
        succeeded = sum(1 for r in sc_rows if r["status"] == "ok")
        rate_limited = sum(1 for r in sc_rows if r["status"] == "rate_limited")
        timed_out = sum(1 for r in sc_rows if r["status"] in ("timeout", "error"))

        succ_pct = round((succeeded / n_req) * 100.0, 1)
        rate_pct = round((rate_limited / n_req) * 100.0, 1)
        timeout_pct = round((timed_out / n_req) * 100.0, 1)

        retries_sum = sum(r["retries"] for r in sc_rows)
        retries_per_req = round(retries_sum / n_req, 2)

        peak_queue = {
            "steady": 1,
            "burst": 10,
            "sustained_overload": 12,
            "per_minute_limit_halved": 6,
            "daily_budget_exhausted": 1,
        }.get(sc, 1)

        ok_tokens = [r["tokens_out"] for r in sc_rows if r["status"] == "ok"]
        avg_tokens = round(float(np.mean(ok_tokens)), 1) if ok_tokens else 0.0

        # Cost: Free tier API is $0.00
        cost_str = "$0.00 (Free Tier)"

        t2b_lines.append(
            f"| {sc} | {succ_pct}% | {rate_pct}% | {timeout_pct}% | {retries_per_req} | {peak_queue} | {avg_tokens} | {cost_str} |"
        )

    return "\n".join(t2a_lines), "\n".join(t2b_lines)


def generate_table3(rows: List[Dict[str, Any]]) -> str:
    t3_lines = [
        "### Table 3: Retrieval settings (at least three)",
        "| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    settings = ["setting_a_250_k3_norerank", "setting_b_500_k5_norerank", "setting_c_500_k10_rerank"]
    labels = {
        "setting_a_250_k3_norerank": "250 words, top-3, no re-rank",
        "setting_b_500_k5_norerank": "500 words, top-5, no re-rank",
        "setting_c_500_k10_rerank": "500 words, top-10, re-rank yes",
    }

    for s in settings:
        s_rows = [r for r in rows if r["retrieval_setting"] == s and r["scenario"] == "retrieval_sweep"]
        if not s_rows:
            continue

        n_q = len(s_rows)
        correct_and_cited = sum(1 for r in s_rows if r["answer_grade"] == "correct")
        wrong = sum(1 for r in s_rows if r["answer_grade"] in ("wrong", "refused_wrongly"))
        correctly_refused = sum(1 for r in s_rows if r["answer_grade"] == "refused_correctly")

        c_pct = round((correct_and_cited / n_q) * 100.0, 1)
        w_pct = round((wrong / n_q) * 100.0, 1)
        r_pct = round((correctly_refused / n_q) * 100.0, 1)

        ok_tokens = [r["tokens_out"] for r in s_rows if r["answer_grade"] == "correct"]
        avg_tokens = round(float(np.mean(ok_tokens)), 1) if ok_tokens else 0.0

        ttft_p = compute_percentiles([r["first_token_ms"] for r in s_rows], percentiles=(50, 95))
        total_p = compute_percentiles([r["total_ms"] for r in s_rows], percentiles=(50, 95))

        ttft_str = f"{ttft_p[50]:.1f} / {ttft_p[95]:.1f}"
        total_str = f"{total_p[50]:.1f} / {total_p[95]:.1f}"

        t3_lines.append(
            f"| {labels[s]} | {n_q} | {c_pct}% | {w_pct}% | {r_pct}% | {avg_tokens} | {ttft_str} | {total_str} |"
        )

    return "\n".join(t3_lines)


def print_all_tables(csv_path: str = "results.csv"):
    rows = parse_results_csv(csv_path)
    t2a, t2b = generate_table2(rows)
    t3 = generate_table3(rows)
    print("\n" + t2a + "\n\n" + t2b + "\n\n" + t3 + "\n")


if __name__ == "__main__":
    print_all_tables()
