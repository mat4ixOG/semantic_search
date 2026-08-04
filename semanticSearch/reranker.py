from sentence_transformers import  CrossEncoder
import  config

_model = None


def get_reranker():
    global _model

    if _model is None:
        _model = CrossEncoder(config.CROSS_ENCODER_MODEL)

    return _model


def score_pairs(query: str, texts: list[str]) -> list[float]:
    """
    Relevance score of every text against the query, in one batched pass.
    Shared with the compressor so it does not load a second model.
    """

    if not texts:
        return []

    pairs = [
        (query, text)
        for text in texts
    ]

    return [float(score) for score in get_reranker().predict(pairs)]


def rerank(question: str, hits: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank retrieved chunks using a Cross Encoder.

    Args:
        question: User query.
        hits: Retrieved chunks from the search pipeline.
        top_k: Number of chunks to return after reranking.

    Returns:
        Top-k reranked hits.
    """

    if not hits:
        return []

    scores = score_pairs(question, [hit["text"] for hit in hits])

    # Attach scores without modifying the original objects
    scored_hits = []

    for hit, score in zip(hits, scores):
        current_hit = hit.copy()
        current_hit["rerank_score"] = score
        scored_hits.append(current_hit)

    # Sort by relevance score (highest first)
    scored_hits.sort(
        key=lambda hit: hit["rerank_score"],
        reverse=True,
    )

    return scored_hits[:top_k]
