import re

import config
from llm import UTILITY_MODEL, ask_llm
from prompts import compression_messages
from reranker import score_pairs

NOT_RELEVANT = "NOT_RELEVANT"

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


def compress(question: str, hits: list[dict]) -> list[dict]:
    """
    Contextual compression.

    Reranking picks the right chunks, but a chunk is 800 characters of raw
    page text and most of it is noise. Keep only the sentences that answer
    the question, and drop chunks that turn out to be irrelevant.

    Falls back to the untouched hits whenever compression would leave the
    answer with nothing to work from.
    """

    if not config.USE_COMPRESSION or not hits:
        return hits

    if config.COMPRESSION_MODE == "llm":
        compressed_hits = compress_with_llm(question, hits)
    else:
        compressed_hits = compress_with_cross_encoder(question, hits)

    if not compressed_hits:
        return hits

    return compressed_hits


def split_sentences(text: str) -> list[str]:
    """
    PDF text arrives with line breaks in the middle of sentences and short
    heading fragments, so anything below MIN_SENTENCE_LENGTH is glued to the
    next piece instead of being scored on its own.
    """

    normalised = _WHITESPACE.sub(" ", text).strip()

    if not normalised:
        return []

    sentences = []
    buffer = ""

    for part in _SENTENCE_BOUNDARY.split(normalised):
        buffer = f"{buffer} {part}".strip()

        if len(buffer) >= config.MIN_SENTENCE_LENGTH:
            sentences.append(buffer)
            buffer = ""

    if buffer:
        if sentences:
            sentences[-1] = f"{sentences[-1]} {buffer}"
        else:
            sentences.append(buffer)

    return sentences


def compress_with_cross_encoder(question: str, hits: list[dict]) -> list[dict]:
    """
    Score every sentence with the reranker that is already loaded. All
    sentences of all chunks go through in one batch, so this costs
    milliseconds and no LLM call.
    """

    sentences_per_hit = [split_sentences(hit["text"]) for hit in hits]

    flat_sentences = [
        sentence
        for sentences in sentences_per_hit
        for sentence in sentences
    ]

    scores = score_pairs(question, flat_sentences)

    compressed_hits = []
    cursor = 0

    for hit, sentences in zip(hits, sentences_per_hit):
        hit_scores = scores[cursor:cursor + len(sentences)]
        cursor += len(sentences)

        if not sentences:
            continue

        best_score = max(hit_scores)

        if best_score < config.COMPRESSION_DROP_SCORE:
            continue

        kept = [
            sentence
            for sentence, score in zip(sentences, hit_scores)
            if score >= config.COMPRESSION_MIN_SCORE
        ]

        # Nothing cleared the bar but the chunk is worth keeping, so keep the
        # single best sentence rather than an empty excerpt.
        if not kept:
            kept = [sentences[hit_scores.index(best_score)]]

        current_hit = hit.copy()
        current_hit["original_text"] = hit["text"]
        current_hit["text"] = " ".join(kept)
        current_hit["compressed"] = len(kept) < len(sentences)
        current_hit["compression_score"] = best_score

        compressed_hits.append(current_hit)

    return compressed_hits


def compress_with_llm(question: str, hits: list[dict]) -> list[dict]:
    """
    Ask the model to copy out the relevant sentences. Better at ignoring
    sentences that merely share vocabulary with the question, at the cost of
    one LLM call per chunk.
    """

    compressed_hits = []

    for hit in hits:
        messages = compression_messages(question, hit["text"])

        extract = ask_llm(
            messages=messages,
            pref_model=UTILITY_MODEL,
        )

        extract = (extract or "").strip()

        if not extract or NOT_RELEVANT in extract.upper():
            continue

        current_hit = hit.copy()
        current_hit["original_text"] = hit["text"]

        # A longer "extract" means the model padded it instead of copying,
        # so keep the original text for that chunk.
        if len(extract) <= len(hit["text"]):
            current_hit["text"] = extract
            current_hit["compressed"] = True
        else:
            current_hit["compressed"] = False

        compressed_hits.append(current_hit)

    return compressed_hits
