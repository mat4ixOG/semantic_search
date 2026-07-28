from embedding_model import get_model
from vector_store import collection

model = get_model()


def search(query, top_k=3):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    hits = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": text,
            "page": metadata["page"],
            "document": metadata["document"],
            "distance": distance,
        })

    return hits
