from pathlib import Path

import chromadb
import config

DB_PATH_NAME = config.CHROMA_DB_PATH
COLLECTION_NAME = config.COLLECTION_NAME

DB_PATH = Path(__file__).resolve().parent / DB_PATH_NAME

client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


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

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(ids)
