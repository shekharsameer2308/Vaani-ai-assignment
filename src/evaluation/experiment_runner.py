from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import os
import random
import time
from typing import Dict, List, Any, Optional

from src.evaluation.benchmark_questions import BENCHMARK_QUESTIONS, BenchmarkQuestion
from src.pipeline.pipeline import GroundedRAGPipeline, PipelineResult, REFUSAL_MESSAGE
from src.retrieval.retriever import get_retriever_for_setting
from src.retrieval.settings import RETRIEVAL_SETTINGS
from src.models.standin_service import StandInModelService
from src.models.groq_client import GroqClient
from src.models.base import BaseLLMClient


RESULTS_CSV_HEADERS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
]


class ExperimentRunner:
    """Automated simulation engine executing load shapes, retrieval sweeps, and results logging."""

    def __init__(self, output_csv: str = "results.csv", clear_on_start: bool = False):
        self.output_csv = output_csv
        if clear_on_start or not os.path.exists(self.output_csv):
            self.clear_csv()

    def clear_csv(self):
        """Reset results.csv with header."""
        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(RESULTS_CSV_HEADERS)

    def log_result(
        self,
        request_id: str,
        scenario: str,
        source: str,
        start_time: str,
        res: PipelineResult,
    ):
        """Append a record to results.csv."""
        with open(self.output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                request_id,
                scenario,
                source,
                start_time,
                res.first_token_ms,
                res.total_ms,
                res.status,
                res.retries,
                res.tokens_in,
                res.tokens_out,
                res.retrieval_setting,
                res.answer_grade,
            ])

    def run_single_request(
        self,
        pipeline: GroundedRAGPipeline,
        question: BenchmarkQuestion,
        scenario: str,
        source: str,
        req_index: int,
    ) -> PipelineResult:
        """Run a single request through the pipeline and log it."""
        req_id = f"{scenario}_{source}_{req_index:04d}"
        iso_start = datetime.utcnow().isoformat() + "Z"
        res = pipeline.run(question.query, expected_answerable=question.expected_answerable)
        self.log_result(req_id, scenario, source, iso_start, res)
        return res

    def run_retrieval_sweep(self):
        """Run all 3 retrieval settings across the 15 benchmark questions."""
        for setting_name in RETRIEVAL_SETTINGS:
            client = StandInModelService(requests_per_minute=200, tokens_per_minute=200000)
            retriever = get_retriever_for_setting(setting_name)
            pipeline = GroundedRAGPipeline(retriever=retriever, llm_client=client, setting_name=setting_name)

            for idx, q in enumerate(BENCHMARK_QUESTIONS, start=1):
                self.run_single_request(
                    pipeline=pipeline,
                    question=q,
                    scenario="retrieval_sweep",
                    source="stand-in",
                    req_index=idx,
                )

    def run_load_scenarios(self):
        """Execute the five mandatory load shapes: steady, burst, sustained_overload, limit_halved, daily_exhausted."""
        retriever = get_retriever_for_setting("setting_c_500_k10_rerank")
        questions = BENCHMARK_QUESTIONS

        # 1. Steady Load Scenario (20 requests, spaced evenly at 20 req/min = 3.0s delay)
        standin_steady = StandInModelService(requests_per_minute=30)
        pipe_steady = GroundedRAGPipeline(retriever, standin_steady, "setting_c_500_k10_rerank")
        for i in range(20):
            q = questions[i % len(questions)]
            self.run_single_request(pipe_steady, q, scenario="steady", source="stand-in", req_index=i + 1)
            time.sleep(0.15)  # Accelerated simulation time

        # 2. Burst Scenario (25 concurrent requests dispatched simultaneously)
        standin_burst = StandInModelService(requests_per_minute=30)
        pipe_burst = GroundedRAGPipeline(retriever, standin_burst, "setting_c_500_k10_rerank")
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    self.run_single_request,
                    pipe_burst,
                    questions[i % len(questions)],
                    "burst",
                    "stand-in",
                    i + 1,
                )
                for i in range(25)
            ]
            for f in as_completed(futures):
                f.result()

        # 3. Sustained Overload Scenario (40 requests at high concurrency exceeding RPM)
        standin_overload = StandInModelService(requests_per_minute=30, tokens_per_minute=10000)
        pipe_overload = GroundedRAGPipeline(retriever, standin_overload, "setting_c_500_k10_rerank")
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [
                executor.submit(
                    self.run_single_request,
                    pipe_overload,
                    questions[i % len(questions)],
                    "sustained_overload",
                    "stand-in",
                    i + 1,
                )
                for i in range(40)
            ]
            for f in as_completed(futures):
                f.result()

        # 4. Per-Minute Limit Halved Scenario (RPM halved to 15, 20 requests)
        standin_halved = StandInModelService(requests_per_minute=15, tokens_per_minute=6000)
        pipe_halved = GroundedRAGPipeline(retriever, standin_halved, "setting_c_500_k10_rerank")
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(
                    self.run_single_request,
                    pipe_halved,
                    questions[i % len(questions)],
                    "per_minute_limit_halved",
                    "stand-in",
                    i + 1,
                )
                for i in range(20)
            ]
            for f in as_completed(futures):
                f.result()

        # 5. Daily Budget Exhausted Scenario (15 requests arriving after budget is exhausted)
        standin_exhausted = StandInModelService()
        standin_exhausted.exhaust_daily_budget()
        pipe_exhausted = GroundedRAGPipeline(retriever, standin_exhausted, "setting_c_500_k10_rerank")
        for i in range(15):
            q = questions[i % len(questions)]
            self.run_single_request(
                pipe_exhausted,
                q,
                scenario="daily_budget_exhausted",
                source="stand-in",
                req_index=i + 1,
            )

    def run_real_service_sample(self, groq_api_key: Optional[str] = None, count: int = 5):
        """Run a modest sample against the real Groq API to record real telemetry."""
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            return

        client = GroqClient(api_key=api_key)
        retriever = get_retriever_for_setting("setting_c_500_k10_rerank")
        pipeline = GroundedRAGPipeline(retriever, client, "setting_c_500_k10_rerank")

        for i in range(min(count, len(BENCHMARK_QUESTIONS))):
            q = BENCHMARK_QUESTIONS[i]
            self.run_single_request(pipeline, q, scenario="steady", source="real", req_index=i + 1)
            time.sleep(2.0)
