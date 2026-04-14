from dotenv import load_dotenv
load_dotenv()
import os

import streamlit as st
from src.document_processor import load_and_chunk_document
from src.embedder import get_embedding, get_embeddings_batch
from src.vector_store import VectorStore
from src.llm import generate_answer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind – AI Document Q&A",
    page_icon="🧠",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

vs: VectorStore = st.session_state.vector_store

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://endee.io/favicon.ico", width=32)
    st.title("DocMind 🧠")
    st.caption("Powered by **Endee** Vector DB + Groq")

    st.markdown("---")
    st.subheader("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("📥 Index Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Processing and indexing documents…"):
                new_docs = []
                for f in uploaded_files:
                    if f.name in st.session_state.indexed_docs:
                        st.info(f"⏭️ Already indexed: {f.name}")
                        continue
                    chunks = load_and_chunk_document(f)
                    if not chunks:
                        st.error(f"Could not parse {f.name}")
                        continue

                    embeddings = get_embeddings_batch([c["text"] for c in chunks])
                    vectors = []
                    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                        vectors.append({
                            "id": f"{f.name}__chunk_{i}",
                            "vector": emb,
                            "meta": {
                                "text": chunk["text"],
                                "source": f.name,
                                "chunk_index": i,
                            },
                            "filter": {"source": f.name},
                        })
                    vs.upsert(vectors)
                    new_docs.append(f.name)
                    st.session_state.indexed_docs.append(f.name)

                if new_docs:
                    st.success(f"✅ Indexed: {', '.join(new_docs)}")

    if st.session_state.indexed_docs:
        st.markdown("**Indexed documents:**")
        for doc in st.session_state.indexed_docs:
            st.markdown(f"- 📄 `{doc}`")

    st.markdown("---")
    if st.button("🗑️ Clear Everything", use_container_width=True):
        vs.delete_index()
        st.session_state.indexed_docs = []
        st.session_state.chat_history = []
        st.session_state.vector_store = VectorStore()
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🧠 DocMind – AI Document Q&A")
st.caption(
    "Upload documents in the sidebar, then ask anything about their contents. "
    "Powered by **Endee** vector database for semantic search + RAG."
)

# Chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📎 Sources used"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']}** (chunk {src['chunk_index']})")
                    st.caption(src["text"][:300] + "…")

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):
    if not st.session_state.indexed_docs:
        st.warning("⬅️ Please upload and index documents first.")
        st.stop()

    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            query_emb = get_embedding(prompt)
            results = vs.query(query_emb, top_k=5)

            if not results:
                answer = "I couldn't find relevant information in the indexed documents."
                sources = []
            else:
                context_chunks = [r["meta"] for r in results]
                answer = generate_answer(prompt, context_chunks)
                sources = context_chunks

        st.markdown(answer)
        if sources:
            with st.expander("📎 Sources used"):
                for src in sources:
                    st.markdown(f"**{src['source']}** (chunk {src['chunk_index']})")
                    st.caption(src["text"][:300] + "…")

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
