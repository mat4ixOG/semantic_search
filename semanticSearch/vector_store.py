from pathlib import Path

import chromadb
import config

DB_PATH = Path(__file__).resolve().parent / config.CHROMA_DB_PATH

_client = None
_collection = None


def get_collection():
    """
    Opened on first use. Connecting to Chroma at import time makes every
    module that touches this one unimportable without a database.
    """

    global _client, _collection

    if _collection is None:
        _client = chromadb.PersistentClient(path=str(DB_PATH))
        _collection = _client.get_or_create_collection(
            name=config.COLLECTION_NAME
        )

    return _collection


def store_embeddings(chunks):
    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in chunks:
        ids.append(f"{chunk['document']}_p{chunk['page']}_{chunk['chunk_id']}")

        documents.append(chunk["text"])

        embeddings.append(chunk["embedding"])

        metadatas.append({
            "document": chunk["document"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"]
        })

    get_collection().upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(ids)


def count() -> int:
    return get_collection().count()
