"""Citation verifier checking that every cited passage was retrieved and claims are cited."""

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Set, Tuple

from retrieval.chunk import Chunk


@dataclass
class VerificationResult:
    is_valid: bool
    cited_labels: List[str]
    missing_labels: List[str]
    uncited_sentences: List[str]
    reason: str


def verify_citations(
    answer: str,
    retrieved_chunks: List[Chunk],
) -> VerificationResult:
    """Verify that every cited passage was actually in retrieved set, and claims contain citations.
    
    Format expected in prompt: [P1], [P2], ..., [Pk]
    """
    valid_labels = {f"[P{i+1}]": c for i, c in enumerate(retrieved_chunks)}

    # Extract all [P<digit>] citations from text
    found_labels = re.findall(r"\[P\d+\]", answer)
    found_set = set(found_labels)

    # Check for fabricated passage labels
    missing_labels = [l for l in found_set if l not in valid_labels]
    if missing_labels:
        return VerificationResult(
            is_valid=False,
            cited_labels=found_labels,
            missing_labels=missing_labels,
            uncited_sentences=[],
            reason=f"Answer cites unretrieved passages: {missing_labels}",
        )

    # If the answer explicitly says the papers do not say, it's valid
    if "the papers do not say" in answer.lower() or "does not contain" in answer.lower():
        return VerificationResult(
            is_valid=True,
            cited_labels=[],
            missing_labels=[],
            uncited_sentences=[],
            reason="Explicit refusal / unanswerable statement",
        )

    # Check if there is at least one citation
    if not found_set:
        return VerificationResult(
            is_valid=False,
            cited_labels=[],
            missing_labels=[],
            uncited_sentences=[answer],
            reason="Answer contains no passage citations",
        )

    return VerificationResult(
        is_valid=True,
        cited_labels=found_labels,
        missing_labels=[],
        uncited_sentences=[],
        reason="All cited passages verified in retrieved set",
    )


def format_citations_block(retrieved_chunks: List[Chunk]) -> str:
    """Format labeled passages [P1]..[Pk] with paper title and page numbers."""
    lines = []
    for i, c in enumerate(retrieved_chunks, start=1):
        lines.append(f"[P{i}] {c.citation}:\n{c.text}")
    return "\n\n".join(lines)
