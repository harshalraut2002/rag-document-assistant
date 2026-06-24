# Document Q&A Assistant (RAG) — Plain-English Run Guide

This is a real Python app (more hands-on than the no-code projects). Just follow
the steps in order. You only need: your Mac, the free Gemini key, and ~20 minutes.

**What it is:** upload documents → ask questions → get answers grounded in the
text, with sources. **What it proves:** real Python + AI engineering (the
"retrieval-augmented generation" technique) — depth the other projects don't show.

---

## What's in this folder
- `app.py` — the app itself (you don't need to edit it)
- `requirements.txt` — the list of things to install
- `sample_docs/laerdal_overview.txt` — a test document to try
- `README.md` — the GitHub write-up
- `HOW_TO_RUN.md` — this guide

---

## Step 1 — Get a free Gemini API key (1 min)
1. Go to **aistudio.google.com** → sign in with Google.
2. Click **Get API key → Create API key**.
3. Copy it somewhere safe. (Same key you used for the Lovable app — you can reuse it.)

## Step 2 — Open Terminal in this folder
1. Open the **Terminal** app (Cmd+Space, type "Terminal", Enter).
2. Type `cd ` (with a space), then **drag this project folder** from Finder into
   the Terminal window — it pastes the path. Press **Enter**.
   You should now be "inside" the `RAG_Document_Assistant` folder.

## Step 3 — Check Python is installed
Type:
```bash
python3 --version
```
- If it shows a version (e.g. `Python 3.11`), you're good.
- If not, install Python from **python.org/downloads** (the big yellow button), then reopen Terminal.

## Step 4 — Install the app's requirements (1–2 min)
```bash
pip3 install -r requirements.txt
```
Wait for it to finish (it downloads Streamlit, Gemini, etc.). Some warnings in
yellow are normal; only red **errors** matter.

## Step 5 — Start the app
```bash
streamlit run app.py
```
Your browser opens automatically at `localhost:8501` with the app.
(If the command isn't found, use `python3 -m streamlit run app.py` instead.)

## Step 6 — Use it
1. In the left sidebar, **paste your Gemini API key**.
2. Click **Browse files** and upload `sample_docs/laerdal_overview.txt`
   (or any PDF/text of your own).
3. Click **Build index** — wait for "Indexed N chunks."
4. In the main box, **ask a question**. Try:
   - "What compression rate does CPR need?"
   - "Where is Laerdal headquartered?"
   - Norwegian: *"Hvor er Laerdal sitt hovedkontor?"*
5. The answer appears with a **"Sources used"** panel showing which passages it used.

## Step 7 — Take your screenshots
Capture: (a) the app with a question + answer, (b) the expanded **Sources used**
panel, and (c) one **Norwegian** question/answer. These go in your documentation.

To stop the app later: go back to Terminal and press **Ctrl + C**.

---

## Step 8 — Put it on GitHub (optional, for your portfolio)
1. Create a new repo on github.com (e.g. `rag-document-assistant`).
2. Upload `app.py`, `requirements.txt`, `README.md`, and the `sample_docs` folder
   (drag-and-drop works on the GitHub website via "Add file → Upload files").
3. **Do NOT upload your API key** — the app asks for it at runtime, so there's no
   key in the code. Good.

---

## If something goes wrong
- **`command not found: streamlit`** → run `python3 -m streamlit run app.py`.
- **`pip3: command not found`** → try `pip install -r requirements.txt`.
- **An error mentioning the Gemini model name** → models occasionally get renamed.
  Open `app.py`, and near the top change `gemini-1.5-flash` to the current flash
  model name from aistudio.google.com (and `text-embedding-004` likewise). That's
  the only spot that would ever need a tweak.
- **"Resource exhausted / quota"** → the free tier hit its rate limit; wait a
  minute and try again.

## Interview talking points (for Niels)
- "I built a RAG document assistant in Python — it embeds documents, retrieves the
  most relevant passages for a question, and has the LLM answer **only** from those,
  with citations."
- **Why RAG:** grounds answers in real documents, works on private/up-to-date data
  the model never saw, and stays traceable — no retraining needed.
- **What I'd add next:** a persistent vector database so the index isn't rebuilt
  each session, and document storage.
