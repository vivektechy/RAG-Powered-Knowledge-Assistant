import os
import glob
import ollama

from pypdf import PdfReader

from app import config
from app.vector_store import SimpleVectorStore


SUPPORTED_EXTENSIONS = (".txt", ".pdf")


def load_pages_from_file(path: str):
    """Load text page-by-page from TXT or PDF."""

    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                pages.append((page_number, text))

        return pages

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:
        return [(1, f.read())]


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int
) -> list[str]:

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

        start += chunk_size - overlap

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Create embeddings using local Ollama."""

    response = ollama.embed(
        model=config.EMBEDDING_MODEL,
        input=texts
    )

    return response["embeddings"]


def get_document_directories():

    project_root = config.PROJECT_ROOT

    directories = [
        project_root / "data" / "sample_docs",
        project_root / "data" / "uploads",
    ]

    return directories


def main():

    store = SimpleVectorStore(
        config.VECTOR_STORE_PATH
    )

    # Rebuild everything from scratch
    store.reset()

    file_paths = []

    for directory in get_document_directories():

        if not directory.exists():
            continue

        for path in directory.rglob("*"):

            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ):
                file_paths.append(path)

    if not file_paths:

        print("No PDF or TXT files found.")

        return

    all_chunks = []
    all_ids = []
    all_metadatas = []

    for path in file_paths:

        print()
        print(f"Reading: {path.name}")

        try:

            pages = load_pages_from_file(
                str(path)
            )

            total_chunks = 0

            for page_number, page_text in pages:

                chunks = chunk_text(
                    page_text,
                    config.CHUNK_SIZE,
                    config.CHUNK_OVERLAP
                )

                for chunk_index, chunk in enumerate(chunks):

                    all_chunks.append(chunk)

                    all_ids.append(
                        f"{path.name}-{page_number}-{chunk_index}"
                    )

                    all_metadatas.append({
                        "source": path.name,
                        "page": page_number,
                        "chunk_index": chunk_index
                    })

                    total_chunks += 1

            print(
                f"  {path.name}: "
                f"{total_chunks} chunks"
            )

        except Exception as e:

            print(
                f"  ERROR reading {path.name}: {e}"
            )

    if not all_chunks:

        print(
            "No readable text was extracted "
            "from the documents."
        )

        return

    print()
    print(
        f"Embedding {len(all_chunks)} chunks total..."
    )

    batch_size = 100

    for start in range(
        0,
        len(all_chunks),
        batch_size
    ):

        end = start + batch_size

        batch_chunks = all_chunks[start:end]
        batch_ids = all_ids[start:end]
        batch_meta = all_metadatas[start:end]

        embeddings = embed_texts(
            batch_chunks
        )

        store.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_chunks,
            metadatas=batch_meta
        )

        print(
            f"  Embedded {min(end, len(all_chunks))}"
            f"/{len(all_chunks)}"
        )

    store.persist()

    print()
    print(
        f"Done. Ingested {len(all_chunks)} chunks "
        f"from {len(file_paths)} files."
    )

    print(
        f"Vector store: "
        f"{config.VECTOR_STORE_PATH}"
    )


if __name__ == "__main__":
    main()