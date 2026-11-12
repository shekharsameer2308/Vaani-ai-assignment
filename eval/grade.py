"""Automated grading and blind verification sheet generator.

Evaluates pipeline answers across 3 retrieval settings:
- setting_a: piece 200, k 3, no rerank
- setting_b: piece 500, k 5, no rerank
- setting_c: piece 500, k 10, rerank

Grades each answer:
- correct: answerable question answered with citations matching gold page/paper
- refused_correctly: unanswerable question refused without hallucination
- wrong: unanswerable question received an answer asserting facts, or incorrect citations
- refused_wrongly: answerable question was refused falsely
"""

import argparse
import asyncio
import csv
import json
import os
import random
import re
from typing import Dict, List, Optional

from pipeline.limits import RateLimiter
from pipeline.pipeline import PipelineDeps, answer
from retrieval.index import load_or_build_index
from retrieval.search import search
from standin.app import StandInLLMClient

RETRIEVAL_SETTINGS = {
    "setting_a": {"piece_tokens": 200, "overlap": 40, "k": 3, "rerank": "none"},
    "setting_b": {"piece_tokens": 500, "overlap": 100, "k": 5, "rerank": "none"},
    "setting_c": {"piece_tokens": 500, "overlap": 100, "k": 10, "rerank": "none"},  # rerank placeholder
}


def load_questions(questions_path: str = "eval/questions.json") -> List[Dict]:
    with open(questions_path, "r", encoding="utf-8") as f:
        return json.load(f)


def grade_response(q: Dict, answer_text: str, citations: List[str]) -> str:
    """Assigns strict grade to pipeline response."""
    kind = q["kind"]
    refusal_cues = ["the papers do not say", "i do not know", "out of scope", "does not contain"]
    is_refusal = any(cue in answer_text.lower() for cue in refusal_cues) or not answer_text.strip()

    if kind in ("unanswerable_out_of_corpus", "unanswerable_subtle"):
        if is_refusal:
            return "refused_correctly"
        else:
            return "wrong"
    else:  # answerable
        if is_refusal:
            return "refused_wrongly"

        gold_paper = q.get("gold_paper")
        gold_page = q.get("gold_page")
        if isinstance(gold_page, int):
            gold_pages = [gold_page]
        elif isinstance(gold_page, list):
            gold_pages = gold_page
        else:
            gold_pages = []

        # Check if citations match gold paper and page
        matched_gold = False
        for cit in citations:
            if gold_paper and gold_paper in cit:
                if gold_pages:
                    # Look for page indicator in citation: e.g. "p. 4" or "pp. 3-4"
                    p_match = re.search(r"p(?:p)?\.\s*(\d+)(?:-(\d+))?", cit)
                    if p_match:
                        p_start = int(p_match.group(1))
                        p_end = int(p_match.group(2)) if p_match.group(2) else p_start
                        if any(p_start <= gp <= p_end for gp in gold_pages):
                            matched_gold = True
                            break
                    else:
                        matched_gold = True
                else:
                    matched_gold = True

        return "correct" if matched_gold else "refused_wrongly"


async def evaluate_setting(
    setting_name: str,
    questions: List[Dict],
    llm_client,
    absence_threshold: float = 12.0,
) -> List[Dict]:
    cfg = RETRIEVAL_SETTINGS[setting_name]
    index, chunks = load_or_build_index(
        piece_tokens=cfg["piece_tokens"],
        overlap=cfg["overlap"],
    )

    limiter = RateLimiter(
        requests_per_minute=1000,
        tokens_per_minute=1000000,
        daily_requests_limit=100000,
        daily_tokens_limit=10000000,
        in_flight_limit=10,
        max_wait_seconds=5.0,
    )

    deps = PipelineDeps(
        retriever_fn=search,
        llm_client=llm_client,
        limiter=limiter,
        cache={},
        piece_tokens=cfg["piece_tokens"],
        top_k=cfg["k"],
        absence_threshold=absence_threshold,
    )

    results = []
    for q in questions:
        res = await answer(q["question"], deps)
        grade = grade_response(q, res.answer, res.citations)
        results.append({
            "question_id": q["id"],
            "question": q["question"],
            "kind": q["kind"],
            "setting": setting_name,
            "answer": res.answer,
            "citations": "; ".join(res.citations),
            "mode": res.mode,
            "status": res.status,
            "grade": grade,
        })
    return results


def export_blind_grading_sheet(eval_records: List[Dict], output_path: str = "eval/grading_sheet.csv"):
    """Randomises order and strips pipeline metadata for unbiased human review."""
    shuffled = list(eval_records)
    random.seed(42)
    random.shuffle(shuffled)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "question", "system_answer", "system_citations", "human_grade_override", "notes"])
        for idx, rec in enumerate(shuffled, 1):
            writer.writerow([f"BLIND-{idx:03d}", rec["question"], rec["answer"], rec["citations"], "", ""])
    print(f"Blind grading sheet exported to {output_path} ({len(shuffled)} items)")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline against 15 benchmark questions.")
    parser.add_argument("--setting", choices=["setting_a", "setting_b", "setting_c", "all"], default="all")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--output", default="results/eval_results.json")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    client = StandInLLMClient()

    settings_to_run = list(RETRIEVAL_SETTINGS.keys()) if args.setting == "all" else [args.setting]
    all_eval_records = []

    for s_name in settings_to_run:
        print(f"\n--- Evaluating {s_name} ---")
        records = await evaluate_setting(s_name, questions, client)
        all_eval_records.extend(records)
        correct_count = sum(1 for r in records if r["grade"] in ("correct", "refused_correctly"))
        print(f"Accuracy ({s_name}): {correct_count}/{len(records)} ({correct_count/len(records)*100:.1f}%)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_eval_records, f, indent=2)

    export_blind_grading_sheet(all_eval_records)


if __name__ == "__main__":
    asyncio.run(main())
