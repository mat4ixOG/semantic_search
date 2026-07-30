from embedding_model import get_model
from vector_store import collection
from rank_bm25 import  BM25Okapi
import numpy as np

model = get_model()

all_chunks = collection.get()
documents = all_chunks["documents"]
tokenized_corpus = [document.lower().split() for document in documents]
bm25 = BM25Okapi(tokenized_corpus)


def search(query, top_k=3):
    vector_hits =  vector_search(query, top_k)
    keyword_hits = keyword_search(query, top_k)

    return merge_results(vector_hits, keyword_hits , top_k)


def vector_search(query, top_k=3):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
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
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    hits = []
    for idx in top_indices:
        hits.append({
            "text": documents[idx],
            "page": all_chunks["metadatas"][idx]["page"],
            "document": all_chunks["metadatas"][idx]["document"],
            "score": scores[idx],
            "id": all_chunks["ids"][idx],
        })

    return hits


def merge_results(vector_hits, keyword_hits, top_k):
    rrf_scores = {}
    hit_map = {}
    k = 60
    for rank , hit in enumerate(vector_hits , start=1):
        chunk_id = hit["id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank)
        hit_map[chunk_id] = hit

    for rank , hit in enumerate(keyword_hits , start=1):
        chunk_id = hit["id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank)
        hit_map[chunk_id] = hit

    sorted_ids = sorted(rrf_scores.items() , key=lambda item: item[1], reverse=True)

    merged = []

    for chunk_id, score in sorted_ids:
        merged.append(hit_map[chunk_id])

    return merged[:top_k]