"""Analysis and table generator for assignment submission.

Parses results.csv (or results/requests.csv) and generates:
- Table 1: Load shape comparison across scenarios
- Table 2a: Cost & latency by retrieval setting
- Table 2b: Retrieval quality (refusal accuracy, citation accuracy)
- Table 3: Fallback behavior under stress
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional
import yaml


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(math.ceil(p / 100.0 * len(sorted_data))) - 1
    return sorted_data[max(0, min(idx, len(sorted_data) - 1))]


def load_pricing(pricing_path: str = "configs/pricing.yaml") -> Dict[str, float]:
    if not os.path.exists(pricing_path):
        return {"prompt_cost_per_m": 0.05, "completion_cost_per_m": 0.08}
    with open(pricing_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    ref = data.get("reference_paid_tier", {})
    return {
        "prompt_cost_per_m": ref.get("cost_per_million_input_tokens", 0.05),
        "completion_cost_per_m": ref.get("cost_per_million_output_tokens", 0.08),
    }


def load_records(csv_path: str = "results.csv") -> List[Dict]:
    if not os.path.exists(csv_path):
        alt = "results/requests.csv"
        if os.path.exists(alt):
            csv_path = alt
        else:
            raise FileNotFoundError(f"Neither {csv_path} nor {alt} exists.")

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "request_id": row["request_id"],
                "scenario": row["scenario"],
                "source": row["source"],
                "start_time": float(row["start_time"]),
                "first_token_ms": float(row["first_token_ms"]),
                "total_ms": float(row["total_ms"]),
                "status": row["status"],
                "retries": int(row["retries"]),
                "tokens_in": int(row["tokens_in"]),
                "tokens_out": int(row["tokens_out"]),
                "retrieval_setting": row["retrieval_setting"],
                "answer_grade": row["answer_grade"],
            })
    return records


def generate_table1(records: List[Dict], pricing: Dict[str, float]) -> str:
    """Table 1: Load shape comparison (scenarios x metrics: throughput, p50/p95/p99 TTFT, error rate, fallback rate, cost)."""
    by_scenario = defaultdict(list)
    for r in records:
        by_scenario[r["scenario"]].append(r)

    rows = []
    scenarios_order = ["steady", "burst", "sustained_overload", "rpm_halved", "daily_exhausted"]
    all_scenarios = [s for s in scenarios_order if s in by_scenario] or list(by_scenario.keys())

    for sc in all_scenarios:
        sc_recs = by_scenario[sc]
        total_reqs = len(sc_recs)
        if total_reqs == 0:
            continue

        start_times = [r["start_time"] for r in sc_recs]
        duration_s = max(1.0, max(start_times) - min(start_times)) if len(start_times) > 1 else 1.0
        throughput_rps = total_reqs / duration_s

        ttft_vals = [r["first_token_ms"] for r in sc_recs if r["first_token_ms"] > 0]
        p50 = percentile(ttft_vals, 50)
        p95 = percentile(ttft_vals, 95)
        p99 = percentile(ttft_vals, 99)

        errors = sum(1 for r in sc_recs if r["status"] in ("error", "timeout", "rate_limited"))
        error_rate = (errors / total_reqs) * 100.0

        fallbacks = sum(1 for r in sc_recs if r["status"] == "rate_limited" or r["answer_grade"] == "refused_correctly")
        fallback_rate = (fallbacks / total_reqs) * 100.0

        # Cost calculation based on reference pricing
        total_in = sum(r["tokens_in"] for r in sc_recs)
        total_out = sum(r["tokens_out"] for r in sc_recs)
        cost_usd = (total_in / 1e6 * pricing["prompt_cost_per_m"]) + (total_out / 1e6 * pricing["completion_cost_per_m"])

        rows.append(
            f"| `{sc}` | {total_reqs} | {throughput_rps:.2f} | {p50:.1f} | {p95:.1f} | {p99:.1f} | {error_rate:.1f}% | {fallback_rate:.1f}% | ${cost_usd:.5f} |"
        )

    header = (
        "### Table 1: Load Shape Comparison Across Traffic Scenarios\n\n"
        "| Scenario | Total Reqs | Throughput (req/s) | p50 TTFT (ms) | p95 TTFT (ms) | p99 TTFT (ms) | Error Rate | Fallback Rate | Est. Cost (USD) |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )
    return header + "\n".join(rows) + "\n"


def generate_table2a(records: List[Dict], pricing: Dict[str, float]) -> str:
    """Table 2a: Cost & latency by retrieval setting."""
    by_setting = defaultdict(list)
    for r in records:
        by_setting[r["retrieval_setting"]].append(r)

    rows = []
    for setting in ["setting_a", "setting_b", "setting_c"]:
        s_recs = by_setting.get(setting, [])
        if not s_recs:
            continue
        n = len(s_recs)
        avg_tokens_in = sum(r["tokens_in"] for r in s_recs) / n
        avg_tokens_out = sum(r["tokens_out"] for r in s_recs) / n

        ttft_vals = [r["first_token_ms"] for r in s_recs if r["first_token_ms"] > 0]
        total_vals = [r["total_ms"] for r in s_recs if r["total_ms"] > 0]
        p50_ttft = percentile(ttft_vals, 50)
        p95_ttft = percentile(ttft_vals, 95)
        p50_total = percentile(total_vals, 50)

        cost_per_req = (avg_tokens_in / 1e6 * pricing["prompt_cost_per_m"]) + (avg_tokens_out / 1e6 * pricing["completion_cost_per_m"])

        rows.append(
            f"| `{setting}` | {n} | {avg_tokens_in:.0f} | {avg_tokens_out:.0f} | {p50_ttft:.1f} | {p95_ttft:.1f} | {p50_total:.1f} | ${cost_per_req:.6f} |"
        )

    header = (
        "### Table 2a: Cost & Latency by Retrieval Setting\n\n"
        "| Retrieval Setting | N | Mean Prompt Tokens | Mean Output Tokens | p50 TTFT (ms) | p95 TTFT (ms) | p50 Total (ms) | Cost / Req (USD) |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )
    return header + "\n".join(rows) + "\n"


def generate_table2b(records: List[Dict]) -> str:
    """Table 2b: Retrieval quality (refusal accuracy, citation accuracy)."""
    by_setting = defaultdict(list)
    for r in records:
        by_setting[r["retrieval_setting"]].append(r)

    rows = []
    for setting in ["setting_a", "setting_b", "setting_c"]:
        s_recs = by_setting.get(setting, [])
        if not s_recs:
            continue
        n = len(s_recs)

        # Citations correct
        correct_citations = sum(1 for r in s_recs if r["answer_grade"] == "correct")
        # Refused correctly
        correct_refusals = sum(1 for r in s_recs if r["answer_grade"] == "refused_correctly")
        # Hallucinations
        hallucinations = sum(1 for r in s_recs if r["answer_grade"] == "hallucinated")
        # False refusals
        false_refusals = sum(1 for r in s_recs if r["answer_grade"] == "refused_incorrectly")

        overall_accuracy = ((correct_citations + correct_refusals) / n) * 100.0

        rows.append(
            f"| `{setting}` | {n} | {correct_citations} | {correct_refusals} | {hallucinations} | {false_refusals} | {overall_accuracy:.1f}% |"
        )

    header = (
        "### Table 2b: Retrieval Quality & Grounding by Setting\n\n"
        "| Retrieval Setting | N | Correct Citations | Correct Refusals | Hallucinations | False Refusals | Overall Grounding Accuracy |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )
    return header + "\n".join(rows) + "\n"


def generate_table3(records: List[Dict]) -> str:
    """Table 3: Fallback behavior under stress."""
    by_scenario = defaultdict(list)
    for r in records:
        by_scenario[r["scenario"]].append(r)

    rows = []
    for sc, sc_recs in by_scenario.items():
        total = len(sc_recs)
        direct_ok = sum(1 for r in sc_recs if r["status"] == "ok")
        rate_limited = sum(1 for r in sc_recs if r["status"] == "rate_limited")
        errors = sum(1 for r in sc_recs if r["status"] == "error")
        total_retries = sum(r["retries"] for r in sc_recs)

        rows.append(
            f"| `{sc}` | {total} | {direct_ok} | {rate_limited} | {errors} | {total_retries} |"
        )

    header = (
        "### Table 3: Fallback and Degradation Behavior Under Stress\n\n"
        "| Scenario | Total Invocations | Direct LLM (OK) | Rate-Limited Fallbacks | Unhandled Errors | Total Retries Executed |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )
    return header + "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark tables from results.csv.")
    parser.add_argument("--csv", default="results.csv")
    parser.add_argument("--pricing", default="configs/pricing.yaml")
    parser.add_argument("--output", default="results/tables.md")
    args = parser.parse_args()

    pricing = load_pricing(args.pricing)
    records = load_records(args.csv)

    t1 = generate_table1(records, pricing)
    t2a = generate_table2a(records, pricing)
    t2b = generate_table2b(records)
    t3 = generate_table3(records)

    full_markdown = f"# Performance & Grounding Analysis Tables\n\n{t1}\n{t2a}\n{t2b}\n{t3}\n"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(full_markdown)

    print(full_markdown)
    print(f"Tables written to {args.output}")


if __name__ == "__main__":
    main()
