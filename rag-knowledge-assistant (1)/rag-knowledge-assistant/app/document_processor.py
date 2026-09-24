import os
import uuid
import ollama

from pypdf import PdfReader

from app import config
from app.vector_store import SimpleVectorStore


def extract_pages(file_path: str):
    """
    Extract text page-by-page from TXT or PDF.
    Returns:
        list of (page_number, text)
    """

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".txt":

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:
            text = f.read()

        return [(1, text)]

    if extension == ".pdf":

        reader = PdfReader(file_path)

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text() or ""

            if text.strip():

                pages.append(
                    (
                        page_number,
                        text
                    )
                )

        return pages

    raise ValueError(
        "Only PDF and TXT files are supported."
    )


def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None
):

    chunk_size = (
        chunk_size
        or config.CHUNK_SIZE
    )

    overlap = (
        overlap
        or config.CHUNK_OVERLAP
    )

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def add_document(
    file_path: str,
    original_filename: str
):

    pages = extract_pages(file_path)

    if not pages:

        raise ValueError(
            "No readable text was found in this file. "
            "If this is a scanned PDF, OCR is required."
        )

    all_chunks = []
    all_metadatas = []

    for page_number, page_text in pages:

        chunks = chunk_text(page_text)

        for chunk in chunks:

            all_chunks.append(chunk)

            all_metadatas.append(
                {
                    "source": original_filename,
                    "page": page_number
                }
            )

    if not all_chunks:

        raise ValueError(
            "Could not create any text chunks."
        )

    response = ollama.embed(
        model=config.EMBEDDING_MODEL,
        input=all_chunks
    )

    embeddings = response["embeddings"]

    store = SimpleVectorStore(
        config.VECTOR_STORE_PATH
    )

    ids = [
        f"{original_filename}-"
        f"{uuid.uuid4().hex[:8]}-{i}"
        for i in range(len(all_chunks))
    ]

    store.add(
        ids=ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadatas
    )

    store.persist()

    return {
        "filename": original_filename,
        "chunks": len(all_chunks)
    }