from chunking import chunk_text
from embeddings import create_embeddings
from pdf_ingestion import DEFAULT_PDF, load_pdf
from vector_store import store_embeddings

CHUNK_SIZE = 800
OVERLAP = 100


def ingest(pdf_path=DEFAULT_PDF, document_name=None):
    document_name = document_name or pdf_path.stem

    pages = load_pdf(pdf_path)
    print(f"loaded {len(pages)} pages from {pdf_path.name}")

    chunks = []
    for page_data in pages:
        chunks.extend(
            chunk_text(page_data, CHUNK_SIZE, OVERLAP, document_name)
        )
    print(f"created {len(chunks)} chunks")

    chunks = create_embeddings(chunks)
    print("embedded chunks")

    stored = store_embeddings(chunks)
    print(f"stored {stored} chunks")


if __name__ == "__main__":
    ingest()
