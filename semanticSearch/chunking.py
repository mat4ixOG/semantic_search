def chunk_text(page_data, chunk_size, overlap, document_name):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    text = page_data["text"]
    page = page_data["page"]

    start = 0
    chunk_id = 1
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start:start + chunk_size]

        if chunk.strip():
            chunks.append({
                "document": document_name,
                "page": page,
                "chunk_id": chunk_id,
                "text": chunk
            })
            chunk_id += 1

        start += step

    return chunks
