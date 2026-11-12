"""Load testing driver and open-loop arrival generator.

Executes traffic scenarios against the RAG pipeline using a thread-safe / coroutine-safe
asyncio.Queue background writer to prevent file write race conditions.

Scenarios supported:
- steady: constant arrival rate within limits
- burst: step-up spike to 3x rate limit
- sustained_overload: persistent 1.5x rate limit
- rpm_halved: mid-run capacity cut from 30 to 15 RPM
- daily_exhausted: trip daily budget breaker
"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import subprocess
import time
import uuid
from typing import Dict, List, Optional
import yaml

from eval.grade import RETRIEVAL_SETTINGS, grade_response, load_questions
from pipeline.limits import RateLimiter
from pipeline.pipeline import PipelineDeps, answer
from retrieval.index import load_or_build_index
from retrieval.search import search
from standin.app import StandInLLMClient
from standin.simulator import StandInSimulator


def compute_file_hash(path: str) -> str:
    if not os.path.exists(path):
        return "none"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def get_git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()[:8]
    except Exception:
        return "unknown"


class CSVBackgroundWriter:
    """Single queue consumer task writing to disk to prevent coroutine race conditions."""

    def __init__(self, requests_csv_paths: List[str], timings_csv_path: str, dry_run: bool = False):
        self.requests_paths = requests_csv_paths
        self.timings_path = timings_csv_path
        self.dry_run = dry_run
        self.queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self):
        if not self.dry_run:
            for p in self.requests_paths:
                os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
                if not os.path.exists(p) or os.path.getsize(p) == 0:
                    with open(p, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "request_id", "scenario", "source", "start_time",
                            "first_token_ms", "total_ms", "status", "retries",
                            "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"
                        ])

            os.makedirs(os.path.dirname(os.path.abspath(self.timings_path)), exist_ok=True)
            if not os.path.exists(self.timings_path) or os.path.getsize(self.timings_path) == 0:
                with open(self.timings_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["request_id", "stage", "start_ms", "end_ms"])

        self._worker_task = asyncio.create_task(self._worker())

    async def _worker(self):
        while True:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                break

            record_type, data = item
            if not self.dry_run:
                if record_type == "request":
                    for p in self.requests_paths:
                        with open(p, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                data["request_id"],
                                data["scenario"],
                                data["source"],
                                f"{data['start_time']:.4f}",
                                f"{data['first_token_ms']:.2f}",
                                f"{data['total_ms']:.2f}",
                                data["status"],
                                data["retries"],
                                data["tokens_in"],
                                data["tokens_out"],
                                data["retrieval_setting"],
                                data["answer_grade"],
                            ])
                elif record_type == "timing":
                    with open(self.timings_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            data["request_id"],
                            data["stage"],
                            f"{data['start_ms']:.2f}",
                            f"{data['end_ms']:.2f}",
                        ])
            self.queue.task_done()

    async def emit_request(self, record: Dict):
        await self.queue.put(("request", record))

    async def emit_timing(self, record: Dict):
        await self.queue.put(("timing", record))

    async def close(self):
        await self.queue.put(None)
        if self._worker_task:
            await self._worker_task


async def run_scenario(
    scenario_name: str,
    scenario_cfg: Dict,
    questions: List[Dict],
    deps: PipelineDeps,
    source: str,
    retrieval_setting: str,
    writer: CSVBackgroundWriter,
    dry_run: bool = False,
):
    print(f"\n================ Running Scenario: {scenario_name} ================")
    print(f"Description: {scenario_cfg.get('description', '')}")

    duration_s = 5 if dry_run else scenario_cfg.get("duration_seconds", 30)
    rate_rps = scenario_cfg.get("rate_rps", 0.5)
    pattern = scenario_cfg.get("arrival_pattern", "poisson")

    if scenario_cfg.get("exhaust_daily", False):
        if hasattr(deps.llm_client, "simulator"):
            deps.llm_client.simulator.exhaust_daily_budget()
        deps.limiter.mark_daily_exhausted()

    start_scenario_time = time.time()
    req_index = 0

    async def dispatch_one(req_num: int, q_item: Dict):
        req_id = f"{scenario_name[:4]}-{uuid.uuid4().hex[:6]}"
        t_req_start = time.time()

        try:
            res = await answer(q_item["question"], deps)
            grade = grade_response(q_item, res.answer, res.citations)

            # Record timings
            prev_ts = 0.0
            for st in res.trace:
                await writer.emit_timing({
                    "request_id": req_id,
                    "stage": st.stage,
                    "start_ms": prev_ts,
                    "end_ms": st.timestamp_ms,
                })
                prev_ts = st.timestamp_ms

            await writer.emit_request({
                "request_id": req_id,
                "scenario": scenario_name,
                "source": source,
                "start_time": t_req_start,
                "first_token_ms": res.first_token_ms,
                "total_ms": res.total_ms,
                "status": res.status,
                "retries": res.retries,
                "tokens_in": res.tokens_in,
                "tokens_out": res.tokens_out,
                "retrieval_setting": retrieval_setting,
                "answer_grade": grade,
            })
        except Exception as e:
            await writer.emit_request({
                "request_id": req_id,
                "scenario": scenario_name,
                "source": source,
                "start_time": t_req_start,
                "first_token_ms": 0.0,
                "total_ms": 0.0,
                "status": "error",
                "retries": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "retrieval_setting": retrieval_setting,
                "answer_grade": "refused_wrongly",
            })

    tasks = []
    while (time.time() - start_scenario_time) < duration_s:
        elapsed = time.time() - start_scenario_time

        # Check dynamic limit drop for rpm_halved
        if scenario_name == "rpm_halved" and elapsed >= scenario_cfg.get("drop_at_seconds", 15):
            halved_limit = scenario_cfg.get("halved_rpm_limit", 15)
            deps.limiter.rpm_limit = halved_limit
            if hasattr(deps.llm_client, "simulator"):
                deps.llm_client.simulator.set_limits(rpm=halved_limit)

        q_item = questions[req_index % len(questions)]
        req_index += 1
        tasks.append(asyncio.create_task(dispatch_one(req_index, q_item)))

        if dry_run and req_index >= 5:
            break

        # Calculate arrival interval
        if pattern == "poisson":
            interval = random.expovariate(rate_rps)
        elif pattern == "step":
            if elapsed < scenario_cfg.get("warmup_seconds", 10):
                interval = 1.0 / rate_rps
            else:
                interval = 1.0 / scenario_cfg.get("spike_rate_rps", 1.5)
        else:  # fixed
            interval = 1.0 / rate_rps

        await asyncio.sleep(min(interval, 1.0))

    if tasks:
        await asyncio.gather(*tasks)
    print(f"Scenario {scenario_name} finished. Total dispatched: {req_index}")


async def main():
    parser = argparse.ArgumentParser(description="Load generator and scenario runner.")
    parser.add_argument("--scenario", choices=["steady", "burst", "sustained_overload", "rpm_halved", "daily_exhausted", "all"], default="all")
    parser.add_argument("--source", choices=["stand-in", "real", "standin", "groq"], default="stand-in")
    parser.add_argument("--retrieval-setting", choices=["setting_a", "setting_b", "setting_c"], default="setting_a")
    parser.add_argument("--dry-run", action="store_true", help="Run 5 requests only without modifying records.")
    parser.add_argument("--scenarios-file", default="load/scenarios.yaml")
    args = parser.parse_args()

    with open(args.scenarios_file, "r", encoding="utf-8") as f:
        scenarios = yaml.safe_load(f)

    questions = load_questions("eval/questions.json")

    # Load BM25 Index for retrieval setting
    r_cfg = RETRIEVAL_SETTINGS[args.retrieval_setting]
    index, chunks = load_or_build_index(
        piece_tokens=r_cfg["piece_tokens"],
        overlap=r_cfg["overlap"],
    )

    # Setup limiter & client
    limiter = RateLimiter(
        requests_per_minute=30,
        tokens_per_minute=14400,
        daily_requests_limit=14400,
        daily_tokens_limit=500000,
        in_flight_limit=5,
        max_wait_seconds=5.0,
    )

    if args.source == "standin":
        sim = StandInSimulator("standin/config.yaml")
        client = StandInLLMClient(simulator=sim)
    else:
        from llm.client import GroqAsyncClient
        client = GroqAsyncClient()

    deps = PipelineDeps(
        retriever_fn=search,
        llm_client=client,
        limiter=limiter,
        cache={},
        piece_tokens=r_cfg["piece_tokens"],
        top_k=r_cfg["k"],
        absence_threshold=12.0,
    )

    requests_paths = ["results/requests.csv", "results.csv"]
    timings_path = "results/timings.csv"

    # Reset or clear results if not dry run and running all
    if not args.dry_run and args.scenario == "all":
        for p in requests_paths:
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists(timings_path):
            os.remove(timings_path)

    writer = CSVBackgroundWriter(requests_paths, timings_path, dry_run=args.dry_run)
    await writer.start()

    run_meta = {
        "start_time": time.time(),
        "git_sha": get_git_sha(),
        "config_hash": compute_file_hash("standin/config.yaml"),
        "limits_hash": compute_file_hash("configs/limits.yaml"),
        "source": args.source,
        "retrieval_setting": args.retrieval_setting,
    }

    try:
        scenarios_to_run = list(scenarios.keys()) if args.scenario == "all" else [args.scenario]
        for sc in scenarios_to_run:
            await run_scenario(
                scenario_name=sc,
                scenario_cfg=scenarios[sc],
                questions=questions,
                deps=deps,
                source=args.source,
                retrieval_setting=args.retrieval_setting,
                writer=writer,
                dry_run=args.dry_run,
            )
    finally:
        await writer.close()
        await client.close()

    if not args.dry_run:
        run_meta["end_time"] = time.time()
        with open("results/run.json", "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)
        print("Experiment run completed. Metadata saved to results/run.json")


if __name__ == "__main__":
    asyncio.run(main())
