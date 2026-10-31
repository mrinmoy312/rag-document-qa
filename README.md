# RAG-Powered Document Q&A

Upload a PDF or text document, ask questions about it in plain English, and get answers
grounded in the document itself — with the exact source chunks cited alongside every answer.

Built with LangChain (LCEL), FAISS, and a swappable LLM backend (OpenAI or Hugging Face).

## How it works

```
 Document (PDF/TXT) --> Chunk & embed --> Vector store (FAISS)
                                                |
                                                v
 Your question --------------------------> Retrieve top-k chunks --> LLM --> Grounded answer
```

Documents are chunked and embedded once, at upload time. Every question re-embeds just the
question, searches the same FAISS index for the closest chunks, and asks the LLM to answer
**using only that retrieved context** — which is what keeps answers grounded instead of
hallucinated, and is what makes RAG different from just chatting with a raw LLM.

## Features

- Drag-and-drop PDF or TXT upload, chunked and embedded in a few seconds
- Chat-style Q&A with full conversation history
- Every answer shows the exact source chunks (with page numbers) it was grounded in
- LLM backend is a one-line config swap: `LLM_PROVIDER=openai` or `huggingface` — same code either way
- FAISS index persists to disk, so you don't re-embed the same document on every restart
- Unit + integration tests that run with no API key and no network access

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | LangChain (LCEL) |
| Vector store | FAISS |
| Embeddings | OpenAI `text-embedding-3-small` **or** local `sentence-transformers` |
| LLM | OpenAI `gpt-5-mini` **or** Hugging Face Inference (`Mistral-7B-Instruct`) |
| UI | Streamlit |
| Tests | pytest, with fake embeddings/LLM so CI needs no secrets |

## Project structure

```
rag-document-qa/
├── app.py                   # Streamlit UI — entry point
├── src/
│   ├── config.py             # All settings, read from .env
│   ├── document_loader.py    # PDF/TXT loading + chunking
│   ├── llm_provider.py       # OpenAI <-> Hugging Face provider switch
│   ├── vector_store.py       # FAISS build / save / load
│   └── rag_pipeline.py       # The LCEL RAG chain itself
├── tests/
│   └── test_pipeline.py      # No API key needed — uses fake embeddings/LLM
├── sample_docs/
│   └── sample.txt            # A short doc about RAG, for a first test run
├── requirements.txt
└── .env.example
```

## Setup

**Requirements:** Python 3.10+

**1. Get the code and create a virtual environment**

```bash
cd rag-document-qa
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure your LLM backend**

```bash
cp .env.example .env
```

Open `.env` and set `LLM_PROVIDER` to one of:

- **`openai`** (recommended default — simplest, most reliable, cheap): get a key at
  [platform.openai.com/api-keys](https://platform.openai.com/api-keys), add billing (a few
  cents covers the whole project), paste it into `OPENAI_API_KEY`.
- **`huggingface`** (free): get a token at
  [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (no billing info
  needed), paste it into `HUGGINGFACEHUB_API_TOKEN`. Embeddings run locally on your machine
  via `sentence-transformers` — the first run downloads a small model (~90MB).

**4. Run it**

```bash
streamlit run app.py
```

Streamlit opens the app in your browser (usually `http://localhost:8501`). Upload
`sample_docs/sample.txt` (or any PDF) and start asking questions.

## Running the tests

```bash
pytest tests/ -v
```

These tests use a deterministic fake embedding model and a fake LLM, so they run in
seconds with no API key and no network access — the kind of test suite you'd actually
want in CI.

## Configuration reference

Everything below lives in `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `huggingface`, or `grok` |
| `OPENAI_CHAT_MODEL` | `gpt-5-mini` | Swap for any OpenAI chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | |
| `HF_CHAT_MODEL` | `mistralai/Mistral-7B-Instruct-v0.3` | Any instruct model on the HF Hub |
| `HF_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Runs locally, free — also powers embeddings when `LLM_PROVIDER=grok` |
| `GROK_CHAT_MODEL` | `grok-4.3` | xAI has no embeddings API, so this provider only covers generation |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Characters per chunk / overlap between chunks |
| `TOP_K` | `4` | How many chunks are retrieved per question |

**Chunk size trade-off:** smaller chunks give more precise retrieval but less surrounding
context per chunk; larger chunks give more context but dilute the similarity signal and
cost more tokens per query. 1000/200 is a reasonable starting point — tune it against your
own documents if answers feel too narrow or too vague.

## Deploying a live demo (free)

A working link in your resume/portfolio is worth far more than a repo people have to clone
to try. [Streamlit Community Cloud](https://streamlit.io/cloud) is free and takes about five
minutes:

1. Push this project to a public GitHub repo (`.env` is already gitignored — never commit it).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **New app**.
3. Point it at your repo, branch `main`, main file `app.py`.
4. Under **Advanced settings → Secrets**, paste the contents of your `.env` file (this is
   Streamlit Cloud's equivalent of environment variables — it's not committed to git).
5. Deploy. You get a public `https://your-app.streamlit.app` URL to put on your resume.

## Possible extensions

Good talking points for an interview, and genuine next steps if you want to keep building:

- **Hybrid search** — combine FAISS similarity search with BM25 keyword search for queries
  with exact terms (names, IDs, section numbers) that pure embedding similarity can miss.
- **Re-ranking** — run retrieved chunks through a cross-encoder re-ranker before generation
  to improve precision on the top-k.
- **Multi-document corpora** — index many documents at once and filter retrieval by source
  document via FAISS metadata filtering.
- **Streaming responses** — swap `StrOutputParser` invocation for `.stream()` so answers
  render token-by-token in the UI instead of all at once.
- **Evaluation harness** — a small labeled Q&A set per document, scored automatically
  (e.g. with RAGAS) so you can measure the effect of chunk size or top-k changes objectively
  instead of eyeballing it.

## License

MIT — see [LICENSE](LICENSE).
