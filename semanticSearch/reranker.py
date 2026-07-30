from sentence_transformers import  CrossEncoder
import  config

model = CrossEncoder(config.CROSS_ENCODER_MODEL)

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

    # Create (question, document) pairs
    pairs = [
        (question, hit["text"])
        for hit in hits
    ]

    # Predict relevance scores
    scores = model.predict(pairs)

    # Attach scores without modifying the original objects
    scored_hits = []

    for hit, score in zip(hits, scores):
        current_hit = hit.copy()
        current_hit["rerank_score"] = float(score)
        scored_hits.append(current_hit)

    # Sort by relevance score (highest first)
    scored_hits.sort(
        key=lambda hit: hit["rerank_score"],
        reverse=True,
    )

    return scored_hits[:top_k]
