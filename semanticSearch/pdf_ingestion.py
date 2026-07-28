from pathlib import Path

import fitz

DEFAULT_PDF = Path(__file__).resolve().parent.parent / "textBook.pdf"


def load_pdf(pdf_path=DEFAULT_PDF):
    doc = fitz.open(pdf_path)

    pages = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pages.append({
            "page": i + 1,
            "text": page.get_text()
        })

    return pages
