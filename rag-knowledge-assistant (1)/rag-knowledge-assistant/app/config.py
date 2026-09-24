import os
from pathlib import Path

from dotenv import load_dotenv


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env if it exists
load_dotenv(PROJECT_ROOT / ".env")


# Local vector store
VECTOR_STORE_PATH = str(
    PROJECT_ROOT
    / os.getenv("CHROMA_DB_DIR", "data/chroma_db")
    / "vector_store.json"
)

# Documents directory
DOCS_DIR = str(
    PROJECT_ROOT
    / os.getenv("DOCS_DIR", "data/sample_docs")
)


# Local Ollama models
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"


# RAG settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 8
RAG_RELEVANCE_THRESHOLD = 0.55