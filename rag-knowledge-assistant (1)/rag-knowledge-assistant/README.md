# Internal Knowledge Assistant (RAG)

Ask questions about internal company documents (leave policy, IT support,
expense policy) and get grounded answers with source citations. Built with
FastAPI, ChromaDB, and OpenAI — deliberately kept lean (no torch /
sentence-transformers) so it installs cleanly on Windows without needing a
C++ compiler.

## How it works
1. `app/ingest.py` reads every `.txt`/`.pdf` in `data/sample_docs/`, splits
   each into overlapping chunks, embeds them with OpenAI, and stores them in
   a local ChromaDB vector database (`data/chroma_db/`).
2. `app/rag_chain.py` embeds your question, retrieves the most relevant
   chunks, and asks the LLM to answer **using only that retrieved context** —
   citing sources, and saying so honestly if the docs don't cover it.
3. `app/main.py` exposes this as a FastAPI endpoint; `ui/streamlit_app.py`
   gives you a chat UI on top of it.

## Setup (Windows, Python 3.11)

```powershell
# 1. Create and activate a virtual environment
py -3.11 -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI key
copy .env.example .env
# then open .env and paste your key from https://platform.openai.com/api-keys

# 4. Ingest the sample documents into the vector store
python -m app.ingest

# 5a. Run the API
uvicorn app.main:app --reload
# then open http://127.0.0.1:8000/docs to try it

# 5b. OR run the chat UI (separate terminal, same venv activated)
streamlit run ui/streamlit_app.py
```

## Try asking
- "How many annual leave days do I get?"
- "How do I reset my password?"
- "What's the per diem for domestic travel?"
- "Can I get reimbursed for alcohol?" (tests that it correctly says no)
- "What's the company's parental leave policy for adoption?" (tests the
  honest "not covered in these docs" fallback, since adoption leave isn't
  in the sample docs)

## Using your own documents
Drop `.txt` or `.pdf` files into `data/sample_docs/` (or change `DOCS_DIR`
in `.env` to point elsewhere), then re-run:
```powershell
python -m app.ingest
```
This rebuilds the vector store from scratch each time, so it's safe to
re-run after adding or editing documents.

## Project structure
```
rag-knowledge-assistant/
├── app/
│   ├── config.py        # env vars, model names, chunk settings
│   ├── ingest.py         # load → chunk → embed → store
│   ├── rag_chain.py       # retrieve → generate, with citations
│   └── main.py            # FastAPI app
├── ui/
│   └── streamlit_app.py    # chat UI
├── data/
│   ├── sample_docs/         # sample internal documents
│   └── chroma_db/            # vector store (created by ingest.py)
├── requirements.txt
└── .env.example
```

## Next steps to make this "enterprise-grade" for a portfolio
- **Hybrid search**: combine this vector search with keyword search (e.g.
  BM25) and merge results — catches exact terms (policy numbers, product
  names) that pure embedding search sometimes misses.
- **Re-ranking**: after retrieving top-K chunks, re-score them with a
  cross-encoder before sending to the LLM — improves precision when you
  have a lot of documents.
- **Feedback loop**: add thumbs up/down on answers, log them (e.g. to
  Supabase or a simple SQLite table), and use that to spot which questions
  the assistant answers badly.
- **Auth**: add an API key or login to the FastAPI endpoints before
  deploying anywhere beyond your own machine.
- **Better chunking**: split by heading/paragraph structure instead of a
  fixed character window, especially once you add longer real documents.
