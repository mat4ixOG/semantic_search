from embedding_model import get_model

model = get_model()


def create_embeddings(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    return chunks
