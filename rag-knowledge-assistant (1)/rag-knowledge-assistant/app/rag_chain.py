import os
import re
import ollama

from app import config
from app.vector_store import SimpleVectorStore


# =========================================================
# ROUTER
# =========================================================

ROUTER_PROMPT = """
You are a routing classifier for a hybrid AI assistant.

Classify the user's question into exactly one category:

DOCUMENT
GENERAL

DOCUMENT means the answer should come from the user's uploaded
PDF/TXT documents.

Examples:
- questions about uploaded documents
- questions about resumes or CVs
- questions about candidates
- questions about people mentioned in uploaded files
- questions about skills, education, experience, projects
- questions about qualifications or certifications
- comparisons between uploaded resumes
- "tell me the names"
- "what skills do they have"
- "compare them"
- "what does the document say"
- "what is the candidate's percentage"
- "what marks did they get"

GENERAL means normal AI knowledge that does not depend on
uploaded documents.

Examples:
- hello
- hi
- how are you
- what is photosynthesis
- explain gravity
- what is Python
- how does machine learning work
- what is the capital of India

Return ONLY:

DOCUMENT

or

GENERAL
"""


# =========================================================
# RAG SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT_RAG = """
You are a document-based knowledge assistant.

Answer the user's question using ONLY the supplied uploaded
document context.

Rules:

1. Use only information found in the supplied context.
2. Never invent names, skills, experience, education, projects,
   qualifications, dates, marks, percentages, or other facts.
3. If the requested information is not present in the context,
   clearly say that it was not found.
4. When multiple documents are provided, use all relevant documents.
5. If the user asks for names, identify names from the documents.
6. If the user asks for a comparison, compare only information
   actually present in the documents.
7. Do not use outside knowledge to fill missing document information.
8. For comparison questions, organize the answer clearly.
9. Keep answers concise but include the relevant facts.
"""


# =========================================================
# GENERAL AI SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT_GENERAL = """
You are a helpful general-purpose AI assistant.

Answer using your general knowledge.

Rules:
- Do not pretend the answer came from uploaded documents.
- Do not mention the document database unless relevant.
- For greetings, respond naturally and briefly.
- If uncertain, say so.
- Keep answers clear and useful.
"""


# =========================================================
# GENERIC DOCUMENT INTENT
# =========================================================

DOCUMENT_INTENT_PATTERNS = [

    r"\bresume\b",
    r"\bresumes\b",
    r"\bcv\b",
    r"\bcvs\b",

    r"\bcandidate\b",
    r"\bcandidates\b",

    r"\buploaded\b",
    r"\bdocument\b",
    r"\bdocuments\b",
    r"\bpdf\b",
    r"\bfile\b",
    r"\bfiles\b",

    r"\bskills?\b",
    r"\bexperience\b",
    r"\bprojects?\b",
    r"\beducation\b",

    r"\bqualification\b",
    r"\bqualifications\b",
    r"\bcertification\b",
    r"\bcertifications\b",

    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bshortlist\b",

    r"\baccording to\b",
    r"\bmentioned\b",
    r"\bprovided\b",

    r"\bwho are\b",
    r"\bhow many\b",

    r"\bboth\b",
    r"\bthese\b",
    r"\bthose\b",
    r"\bthem\b",
    r"\bthey\b",

    # Academic/document facts
    r"\bpercentage\b",
    r"\bpercent\b",
    r"\bmarks?\b",
    r"\bsemester\b",
    r"\bgrade\b",

    # Identity
    r"\bname\b",
    r"\bperson\b",
]


GENERAL_EXACT = {
    "hi",
    "hii",
    "hlo",
    "hello",
    "hey",
    "yo",
    "thanks",
    "thank you",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
}


class RAGAssistant:

    def __init__(self):

        if not os.path.exists(
            config.VECTOR_STORE_PATH
        ):
            raise RuntimeError(
                "No vector store found. "
                "Run `python -m app.ingest` first."
            )

        self.store = SimpleVectorStore(
            config.VECTOR_STORE_PATH
        )

    # =====================================================
    # EMBEDDING
    # =====================================================

    def embed_query(self, query: str):

        response = ollama.embed(
            model=config.EMBEDDING_MODEL,
            input=query
        )

        return response["embeddings"][0]

    # =====================================================
    # RETRIEVAL
    # =====================================================

    def retrieve(
        self,
        query: str,
        top_k: int = None
    ):

        top_k = top_k or config.TOP_K

        query_embedding = self.embed_query(
            query
        )

        results = self.store.query(
            query_embedding,
            query_text=query,
            n_results=top_k
        )

        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        chunks = []

        for document, metadata, distance in zip(
            docs,
            metadatas,
            distances
        ):

            chunks.append({
                "text": document,

                "source": metadata.get(
                    "source",
                    "Unknown document"
                ),

                "page": metadata.get(
                    "page"
                ),

                "score": 1 - distance
            })

        return chunks

    # =====================================================
    # DOCUMENT INTENT
    # =====================================================

    def looks_like_document_question(
        self,
        query: str
    ):

        q = query.lower().strip()

        if q in GENERAL_EXACT:
            return False

        for pattern in DOCUMENT_INTENT_PATTERNS:

            if re.search(pattern, q):
                return True

        return False

    # =====================================================
    # LLM ROUTER
    # =====================================================

    def llm_classify(
        self,
        query: str
    ):

        response = ollama.chat(
            model=config.CHAT_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": ROUTER_PROMPT
                },
                {
                    "role": "user",
                    "content": query
                }
            ],

            options={
                "temperature": 0
            }
        )

        result = (
            response["message"]["content"]
            .strip()
            .upper()
        )

        if result.startswith("DOCUMENT"):
            return "document"

        return "general"

    # =====================================================
    # FINAL CLASSIFICATION
    # =====================================================

    def classify_question(
        self,
        query: str
    ):

        if self.looks_like_document_question(
            query
        ):
            return "document"

        return self.llm_classify(
            query
        )

    # =====================================================
    # GENERAL ANSWER
    # =====================================================

    def general_answer(
        self,
        query: str
    ):

        response = ollama.chat(
            model=config.CHAT_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_GENERAL
                },
                {
                    "role": "user",
                    "content": query
                }
            ],

            options={
                "temperature": 0.5
            }
        )

        return response["message"]["content"]

    # =====================================================
    # DOCUMENT ANSWER
    # =====================================================

    def document_answer(
        self,
        query: str,
        top_k: int = None
    ):

        chunks = self.retrieve(
            query,
            top_k
        )

        # IMPORTANT:
        # Do not throw away retrieved chunks based on
        # the old similarity threshold.
        #
        # Hybrid retrieval has already ranked them.

        relevant_chunks = chunks

        if not relevant_chunks:

            return {
                "answer": (
                    "I couldn't find relevant information "
                    "in the uploaded documents."
                ),

                "sources": [],

                "mode": "document",

                "retrieved_chunks": []
            }

        # =================================================
        # REMOVE DUPLICATES
        # =================================================

        unique_chunks = []

        seen = set()

        for chunk in relevant_chunks:

            key = (
                chunk["source"],
                chunk.get("page"),
                chunk["text"].strip()
            )

            if key not in seen:

                seen.add(key)

                unique_chunks.append(
                    chunk
                )

        # =================================================
        # BUILD CONTEXT
        # =================================================

        context_parts = []

        for chunk in unique_chunks:

            source = chunk["source"]

            page = chunk.get("page")

            if page:
                location = (
                    f"{source}, page {page}"
                )
            else:
                location = source

            context_parts.append(
                f"[Source: {location}]\n"
                f"{chunk['text']}"
            )

        context_blocks = "\n\n".join(
            context_parts
        )

        # =================================================
        # FINAL LLM PROMPT
        # =================================================

        prompt = f"""
Uploaded document context:

{context_blocks}

User question:

{query}
"""

        response = ollama.chat(
            model=config.CHAT_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_RAG
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            options={
                "temperature": 0.1
            }
        )

        answer_text = (
            response["message"]["content"]
            .strip()
        )

        sources = sorted({
            chunk["source"]
            for chunk in unique_chunks
        })

        return {
            "answer": answer_text,
            "sources": sources,
            "mode": "document",
            "retrieved_chunks": unique_chunks
        }

    # =====================================================
    # MAIN ANSWER
    # =====================================================

    def answer(
        self,
        query: str,
        top_k: int = None
    ):

        mode = self.classify_question(
            query
        )

        # GENERAL AI
        if mode == "general":

            answer_text = self.general_answer(
                query
            )

            return {
                "answer": answer_text,
                "sources": [],
                "mode": "general",
                "retrieved_chunks": []
            }

        # DOCUMENT RAG
        return self.document_answer(
            query,
            top_k
        )