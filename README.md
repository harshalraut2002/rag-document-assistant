# Document Q&A Assistant (RAG)

A lightweight **retrieval-augmented generation** app: upload documents, ask
questions in plain English or Norwegian, and get answers grounded in the source
text — with citations back to the exact passages used.

## Features
- Upload PDF / TXT / Markdown documents
- Semantic search over document chunks (embeddings + cosine similarity)
- Grounded answers with **source citations**
- **Bilingual** — answers in the language of the question (English or Norwegian)
- Runs free on the Google Gemini API

## Method — Retrieval-Augmented Generation (RAG)
Rather than sending a question straight to the language model, the app first
splits the uploaded documents into small overlapping chunks, converts each into a
numeric embedding, and stores them. When you ask a question, it embeds the
question too, finds the most semantically similar chunks by cosine similarity, and
passes only those to the model as context.

**Why this approach:** it grounds every answer in your actual documents instead of
the model's memory — which keeps answers accurate, lets the system work on private
or up-to-date material the model was never trained on, and makes each answer
traceable back to its source. It does this with **no model retraining** and runs
entirely on a **free LLM API**, making it cheap, fast, and trustworthy.

## Tech stack
- **Python** + **Streamlit** (UI)
- **Google Gemini** — `text-embedding-004` (embeddings) and `gemini-1.5-flash` (generation)
- **NumPy** for the in-memory vector search
- **pypdf** for PDF text extraction

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then in the app: paste your free Gemini API key (from aistudio.google.com),
upload documents (or use the file in `sample_docs/`), click **Build index**, and
ask questions.

## Pipeline
1. **Extract** text from each uploaded file.
2. **Chunk** it into overlapping word windows (keeps retrieval granular).
3. **Embed** each chunk with Gemini and store the vectors.
4. On a question: **embed the query**, retrieve the top-k most similar chunks.
5. **Generate** an answer from only those chunks, citing their numbers.

## Limitations
Early prototype. In-memory vector store (rebuilds each session) and the Gemini
free tier's rate limits — intended as a demo rather than a production service.
A production version would add a persistent vector database and document storage.

## Author
**Harshal Raut** — epost.harshal@gmail.com
