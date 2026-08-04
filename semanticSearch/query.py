from embedding_model import get_model
from vector_store import get_collection
from rank_bm25 import  BM25Okapi
import numpy as np
import config

# The whole corpus tokenised for BM25. Built on first search rather than on
# import, and rebuildable so an ingest during a server's lifetime is picked
# up instead of being invisible until a restart.
_index = None


def build_index():
    stored = get_collection().get()

    documents = stored["documents"] or []

    return {
        "ids": stored["ids"] or [],
        "documents": documents,
        "metadatas": stored["metadatas"] or [],
        "bm25": BM25Okapi(
            [document.lower().split() for document in documents]
        ) if documents else None,
    }


def get_index():
    global _index

    if _index is None:
        _index = build_index()

    return _index


def refresh_index():
    """
    Call after ingesting. The next search rebuilds from the collection.
    """

    global _index

    _index = None


def search(query, top_k=3):
    vector_hits =  vector_search(query, top_k)
    keyword_hits = keyword_search(query, top_k)

    return merge_results([vector_hits, keyword_hits] , top_k)


def vector_search(query, top_k=3):
    collection = get_collection()

    if not collection.count():
        return []

    query_embedding = get_model().encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    hits = []
    for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
    ):
        hits.append({
            "text": text,
            "page": metadata["page"],
            "document": metadata["document"],
            "distance": distance,
            "id": chunk_id,
        })

    return hits

def keyword_search(query, top_k=3):
    index = get_index()

    if not index["bm25"]:
        return []

    query_tokens = query.lower().split()

    if not query_tokens:
        return []

    scores = index["bm25"].get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    hits = []
    for idx in top_indices:
        # A zero score means the query shares no term with the chunk, so it
        # is only here to pad the list out to top_k.
        if scores[idx] <= 0:
            continue

        hits.append({
            "text": index["documents"][idx],
            "page": index["metadatas"][idx]["page"],
            "document": index["metadatas"][idx]["document"],
            "score": float(scores[idx]),
            "id": index["ids"][idx],
        })

    return hits


def merge_results(hit_lists, top_k):
    """
    Fuse any number of ranked hit lists with Reciprocal Rank Fusion.

    A chunk found by several lists climbs, which is exactly what we want when
    the lists come from different retrievers (vector, keyword) or from
    different phrasings of the same question (multi query, HyDE).
    """

    rrf_scores = {}
    hit_map = {}
    k = config.RRF_K

    for hits in hit_lists:
        for rank , hit in enumerate(hits , start=1):
            chunk_id = hit["id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank)
            hit_map.setdefault(chunk_id, hit)

    sorted_ids = sorted(rrf_scores.items() , key=lambda item: item[1], reverse=True)

    merged = []

    for chunk_id, score in sorted_ids:
        hit = hit_map[chunk_id].copy()
        hit["rrf_score"] = score
        merged.append(hit)

    return merged[:top_k]
