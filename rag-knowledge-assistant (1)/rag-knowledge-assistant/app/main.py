from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.rag_chain import RAGAssistant
from app.document_processor import add_document
from app import config


app = FastAPI(title="Knowledge Assistant")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


assistant = None


def get_assistant():
    global assistant

    if assistant is None:
        assistant = RAGAssistant()

    return assistant


# =========================
# FRONTEND
# =========================

@app.get("/")
def home():
    frontend = Path(config.PROJECT_ROOT) / "frontend" / "index.html"

    if not frontend.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend index.html not found."
        )

    return FileResponse(frontend)


# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# UPLOAD DOCUMENT
# =========================

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in [".pdf", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported."
        )

    upload_dir = Path(config.PROJECT_ROOT) / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(filename).name
    file_path = upload_dir / safe_filename

    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        result = add_document(
            str(file_path),
            safe_filename
        )

        global assistant
        assistant = RAGAssistant()

        return {
            "success": True,
            "message": f"{safe_filename} uploaded successfully.",
            "chunks": result["chunks"]
        }

    except Exception as e:

        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================
# QUERY
# =========================

class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    mode: str


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):

    if not req.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question must not be empty."
        )

    try:

        rag = get_assistant()

        result = rag.answer(
            req.question,
            req.top_k
        )

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "mode": result["mode"]
        }

    except RuntimeError as e:

        raise HTTPException(
            status_code=503,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )