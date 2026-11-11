import argparse
import os
import sys

from src.retrieval.retriever import get_retriever_for_setting
from src.models.standin_service import StandInModelService
from src.models.groq_client import GroqClient
from src.pipeline.pipeline import GroundedRAGPipeline
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.generate_tables import print_all_tables


def main():
    parser = argparse.ArgumentParser(description="The Relay: Grounded arXiv RAG & Load Simulation System")
    parser.add_argument("--query", type=str, help="Run a single question through the pipeline")
    parser.add_argument("--setting", type=str, default="setting_c_500_k10_rerank", help="Retrieval setting name")
    parser.add_argument("--use-real", action="store_true", help="Use real Groq API instead of stand-in service")
    parser.add_argument("--run-experiments", action="store_true", help="Execute all load shapes and retrieval sweeps")
    parser.add_argument("--show-tables", action="store_true", help="Compute and print summary markdown tables from results.csv")

    args = parser.parse_args()

    if args.query:
        client = GroqClient() if args.use_real else StandInModelService()
        retriever = get_retriever_for_setting(args.setting)
        pipeline = GroundedRAGPipeline(retriever=retriever, llm_client=client, setting_name=args.setting)
        result = pipeline.run(args.query)

        print(f"\n--- Query: {result.query} ---")
        print(f"Status: {result.status} | Time: {result.total_ms}ms (TTFT: {result.first_token_ms}ms)")
        print(f"Tokens In: {result.tokens_in} | Tokens Out: {result.tokens_out}")
        print(f"Steps Executed: {' -> '.join(result.steps_executed)}")
        print("\nAnswer:")
        print(result.answer)
        if result.citations:
            print("\nCitations:")
            for c in result.citations:
                print(f"  * {c}")
        return

    if args.run_experiments:
        print("[*] Starting full experimentation suite...")
        runner = ExperimentRunner(output_csv="results.csv", clear_on_start=True)
        print("[*] 1/2 Running retrieval sweeps across 15 benchmark questions...")
        runner.run_retrieval_sweep()
        print("[*] 2/2 Running 5 load scenarios (steady, burst, sustained_overload, halved, daily_exhausted)...")
        runner.run_load_scenarios()
        print("[*] Experiments complete! Generating results tables:\n")
        print_all_tables("results.csv")
        return

    if args.show_tables:
        print_all_tables("results.csv")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
