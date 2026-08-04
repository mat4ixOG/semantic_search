import numpy as np

from embedding_model import get_model
from llm import MODEL_NAME, ask_llm_json
from prompts import (
    FAITHFULNESS_SCHEMA,
    JUDGE_SCHEMA,
    RELEVANCE_SCHEMA,
    correctness_messages,
    faithfulness_messages,
    relevance_messages,
)

CORRECTNESS_POINTS = {
    "correct": 1.0,
    "partially_correct": 0.5,
    "incorrect": 0.0,
}


def retrieval_metrics(hits: list[dict], expected_pages: list[int]) -> dict:
    """
    Deterministic, no LLM. A chunk counts as relevant when it comes from a
    page the answer actually lives on.

    hit_rate        did anything relevant come back at all
    mrr             how high the first relevant chunk ranked
    precision       share of retrieved chunks that were relevant
    recall          share of expected pages that were retrieved
    """

    expected = set(expected_pages or [])

    if not expected:
        return {
            "hit_rate": None,
            "mrr": None,
            "context_precision": None,
            "context_recall": None,
        }

    if not hits:
        return {
            "hit_rate": 0.0,
            "mrr": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }

    relevant_flags = [hit["page"] in expected for hit in hits]

    reciprocal_rank = 0.0
    for rank, is_relevant in enumerate(relevant_flags, start=1):
        if is_relevant:
            reciprocal_rank = 1 / rank
            break

    found_pages = {hit["page"] for hit in hits if hit["page"] in expected}

    return {
        "hit_rate": 1.0 if any(relevant_flags) else 0.0,
        "mrr": reciprocal_rank,
        "context_precision": sum(relevant_flags) / len(relevant_flags),
        "context_recall": len(found_pages) / len(expected),
    }


def semantic_similarity(answer: str, expected: str) -> float:
    """
    Cosine similarity between the two answers using the embedder already in
    memory. Cheap sanity signal, and unlike the LLM judge it is stable
    across runs.
    """

    if not answer or not expected:
        return 0.0

    vectors = get_model().encode([answer, expected])

    first, second = vectors[0], vectors[1]
    norms = np.linalg.norm(first) * np.linalg.norm(second)

    if not norms:
        return 0.0

    return float(np.dot(first, second) / norms)


def faithfulness(answer: str, hits: list[dict]) -> dict:
    """
    Share of the answer's claims that the retrieved context supports.
    This is the hallucination metric.
    """

    if not answer:
        return {"faithfulness": None, "unsupported": []}

    context = "\n\n".join(hit["text"].strip() for hit in hits)

    if not context:
        return {"faithfulness": None, "unsupported": []}

    response = ask_llm_json(
        messages=faithfulness_messages(answer, context),
        schema=FAITHFULNESS_SCHEMA,
        pref_model=MODEL_NAME,
    )

    if not response:
        return {"faithfulness": None, "unsupported": []}

    total = response.get("claims_total") or 0
    supported = response.get("claims_supported") or 0

    # "I don't know" has nothing to hallucinate, so it is not unfaithful.
    if total <= 0:
        return {"faithfulness": 1.0, "unsupported": []}

    supported = max(0, min(supported, total))

    return {
        "faithfulness": supported / total,
        "unsupported": response.get("unsupported") or [],
    }


def correctness(question: str, answer: str, expected: str) -> dict:
    response = ask_llm_json(
        messages=correctness_messages(question, answer, expected),
        schema=JUDGE_SCHEMA,
        pref_model=MODEL_NAME,
    )

    if not response:
        return {"correctness": None, "verdict": None, "reason": None}

    verdict = response.get("verdict")

    return {
        "correctness": CORRECTNESS_POINTS.get(verdict),
        "verdict": verdict,
        "reason": response.get("reason"),
    }


def relevance(question: str, answer: str) -> dict:
    response = ask_llm_json(
        messages=relevance_messages(question, answer),
        schema=RELEVANCE_SCHEMA,
        pref_model=MODEL_NAME,
    )

    if not response:
        return {"answer_relevance": None}

    score = response.get("score")

    if not isinstance(score, int):
        return {"answer_relevance": None}

    # Rescale 1..5 to 0..1 so every metric reads the same way
    return {"answer_relevance": (min(max(score, 1), 5) - 1) / 4}


def average(values) -> float:
    """
    Mean of the values that exist. A metric that could not be computed is
    skipped rather than counted as zero.
    """

    present = [value for value in values if value is not None]

    if not present:
        return None

    return sum(present) / len(present)
