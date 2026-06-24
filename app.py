"""
Document Q&A Assistant (RAG)
----------------------------
A lightweight retrieval-augmented generation app:
  upload documents -> chunk -> embed (Gemini) -> retrieve relevant chunks
  -> answer the question grounded in those chunks, with source citations.

Free to run: uses the Google Gemini API (free tier) for both embeddings and
generation. Answers in the language of the question (English or Norwegian).

Run:  streamlit run app.py
"""

import numpy as np
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader

# Fallback model names (used only if auto-detection fails).
FALLBACK_EMBED = "models/embedding-001"
FALLBACK_GEN = "models/gemini-1.5-flash"


# ---------- model auto-detection (so renames don't break the app) ----------

def pick_models():
    """Ask Google which models this key supports and choose suitable ones."""
    embed_name, gen_candidates = None, []
    for m in genai.list_models():
        methods = getattr(m, "supported_generation_methods", [])
        if "embedContent" in methods and embed_name is None:
            embed_name = m.name
        if "generateContent" in methods:
            gen_candidates.append(m.name)
    gen_name = next((c for c in gen_candidates if "flash" in c.lower()), None)
    if gen_name is None and gen_candidates:
        gen_name = gen_candidates[0]
    return embed_name or FALLBACK_EMBED, gen_name or FALLBACK_GEN


# ---------- core RAG helpers ----------

def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def chunk_text(text: str, size: int = 200, overlap: int = 40):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
        i += size - overlap
    return chunks


def embed_texts(texts, task_type, model):
    vectors = []
    for t in texts:
        resp = genai.embed_content(model=model, content=t, task_type=task_type)
        vectors.append(resp["embedding"])
    return np.array(vectors, dtype=np.float32)


def top_k(query_vec, doc_vecs, k=4):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9)
    sims = d @ q
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]


def answer_question(question, context_chunks, model):
    context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context_chunks))
    prompt = f"""You are a helpful document assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say you don't know based on the provided documents.
Cite the sources you use with their bracket numbers, like [1] or [2].
Detect the language of the question and answer in that same language.

Context:
{context}

Question: {question}

Answer:"""
    return genai.GenerativeModel(model).generate_content(prompt).text


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Document Q&A Assistant (RAG)", page_icon="📄")
st.title("📄 Document Q&A Assistant")
st.caption("Ask questions about your documents — answers are grounded in the text, with sources. English or Norwegian.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Gemini API key", type="password",
                            help="Get a free key at aistudio.google.com")
    files = st.file_uploader("Upload documents (PDF or TXT)",
                             type=["pdf", "txt", "md"], accept_multiple_files=True)
    build = st.button("Build index", type="primary")

if build:
    if not api_key:
        st.sidebar.error("Add your Gemini API key first.")
    elif not files:
        st.sidebar.error("Upload at least one document.")
    else:
        genai.configure(api_key=api_key)
        try:
            embed_model, gen_model = pick_models()
        except Exception as e:
            st.sidebar.error(f"Couldn't reach Gemini. Check your API key. ({e})")
            st.stop()
        with st.spinner("Reading, chunking, and embedding your documents…"):
            all_chunks, sources = [], []
            for f in files:
                for ch in chunk_text(extract_text(f)):
                    all_chunks.append(ch)
                    sources.append(f.name)
            st.session_state.chunks = all_chunks
            st.session_state.sources = sources
            st.session_state.doc_vecs = embed_texts(all_chunks, "retrieval_document", embed_model)
            st.session_state.api_key = api_key
            st.session_state.embed_model = embed_model
            st.session_state.gen_model = gen_model
        st.sidebar.success(f"Indexed {len(all_chunks)} chunks from {len(files)} file(s).")
        st.sidebar.caption(f"Using: {embed_model.split('/')[-1]} + {gen_model.split('/')[-1]}")

if "chunks" in st.session_state:
    question = st.text_input("Ask a question about your documents:")
    if question:
        genai.configure(api_key=st.session_state.api_key)
        with st.spinner("Searching and answering…"):
            q_vec = embed_texts([question], "retrieval_query", st.session_state.embed_model)[0]
            idx, scores = top_k(q_vec, st.session_state.doc_vecs, k=4)
            retrieved = [st.session_state.chunks[i] for i in idx]
            reply = answer_question(question, retrieved, st.session_state.gen_model)

        st.markdown("### Answer")
        st.write(reply)

        with st.expander("Sources used"):
            for rank, i in enumerate(idx):
                st.markdown(f"**[{rank + 1}] {st.session_state.sources[i]}** "
                            f"(similarity {scores[rank]:.2f})")
                st.caption(st.session_state.chunks[i][:400] + "…")
else:
    st.info("👈 Add your Gemini API key, upload documents, and click **Build index** to start.")
