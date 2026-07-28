from pathlib import Path

import chromadb

DB_PATH = Path(__file__).resolve().parent / "chroma_db"

client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_or_create_collection(
    name="employee_handbook"
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
