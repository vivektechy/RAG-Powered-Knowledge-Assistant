"""
Simple chat UI for the RAG assistant.

Run with (from project root):
    streamlit run ui/streamlit_app.py
"""
import sys
from pathlib import Path

# Allow importing the `app` package when Streamlit runs this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from app.rag_chain import RAGAssistant

st.set_page_config(page_title="Internal Knowledge Assistant", page_icon="🤖")
st.title("🤖 Internal Knowledge Assistant")
st.caption("Ask about leave policy, IT support, or expense reimbursement.")


@st.cache_resource
def load_assistant():
    return RAGAssistant()


try:
    assistant = load_assistant()
except RuntimeError as e:
    st.error(str(e))
    st.info("Run `python -m app.ingest` from the project root, then reload this page.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about company policy...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching internal docs..."):
            result = assistant.answer(question)
        st.markdown(result["answer"])
        if result["sources"]:
            st.caption("Sources: " + ", ".join(result["sources"]))

    st.session_state.history.append({"role": "assistant", "content": result["answer"]})
