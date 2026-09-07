# AI JEE Agent v2

> 🚧 **Work in progress.** This is a ground-up rebuild of the [AI JEE Agent](https://huggingface.co/spaces/) — the live demo above is **v1**. v2 is being written by hand, component by component, and is not yet complete. Current state: the full pipeline — extraction → chunking → embeddings → retrieval → reranking → guardrails → LLM integration — runs end to end behind a FastAPI backend; retrieval and prompt evaluation are the remaining gaps (see [Roadmap](#roadmap)).

A retrieval-augmented generation (RAG) system for answering JEE-level Physics and Chemistry questions, grounded in OpenStax textbook content. Every component in v2 is hand-rolled from `transformers` primitives before reaching for frameworks, with a decision log documenting why each choice was made (and what was rejected along the way).

**Why a rebuild?** v1 worked, but not every design decision in it could be defended under questioning. v2's rule: no component ships unless every choice behind it — model, threshold, library, data source — has a reason that survives a "why?" So the pipeline is being rebuilt from scratch, with the reasoning captured in [`Decisions.md`](Decisions.md) as it happens.

**v1 live demo:** [Hugging Face Spaces](https://huggingface.co/spaces/) · **Author:** [Sparsh-25](https://github.com/Sparsh-25)

---

## Architecture

```
PDF (OpenStax)
   │
   ▼
Text Extraction (PyMuPDF + custom junk-line filtering)
   │
   ▼
Chunking (LangChain RecursiveCharacterTextSplitter, token-aware)
   │
   ▼
Embeddings (BGE bi-encoder, asymmetric, MPS on Apple Silicon)
   │
   ▼
Vector Retrieval (NumPy dot product over normalized vectors, threshold = 0.55, top-10)
   │
   ▼
Reranking (bge-reranker-base cross-encoder, top-10 → top-5 scoring above 0)
   │
   ▼
Prompt Construction (structured chunk blocks + grounding rules)
   │
   ▼
LLM (Groq · openai/gpt-oss-120b · 128K context)
```

---

## Pipeline Stages

### 1. Extraction

**Goal:** Clean, layout-aware text with intact formulas and notation.

**Final approach:** PyMuPDF with a custom pre-write filter that strips table of contents, appendices, and junk lines (detected via manual rules + density filtering) before anything hits the `.txt` corpus.

**Data pivot:** Started with NCERT PDFs, which use custom font encodings (CMap references) — every extraction approach returned garbled Unicode. This was confirmed to be a property of the PDFs themselves, not the tooling, by copy-pasting text directly from the PDF into a plain text editor, which produced the same symbol soup. After exhausting seven extraction approaches against NCERT (see Decision Log below), the conclusion was that the corpus was unsalvageable, and the source was switched to **OpenStax University Physics and Chemistry volumes**, where PyMuPDF extracts cleanly on the first pass.

### 2. Chunking

**Final approach:** LangChain's `RecursiveCharacterTextSplitter` with a **token-based length function** (HuggingFace tokenizer, not `len()`), plus a minimum-size filter dropping chunks under 20 tokens. Metadata attached per book/document.

**Why not hand-rolled:** The first manual character splitter produced chunk lengths ranging from 1 to 440 with no overlap and no metadata. Fixable, but re-implementing merge logic, overlap, and metadata handling meant rebuilding LangChain badly.

**Key bug fixed:** Original length function counted characters, not tokens — a 4000-character chunk (~1000 tokens) silently exceeded BGE's input limit.

### 3. Embeddings

- **Model:** BGE (asymmetric variant — query and passage encoded differently), implemented directly from the `transformers` module rather than a wrapper library.
- **Hardware:** MPS acceleration on Apple Silicon (M1).
- **Retrieval threshold:** Cosine similarity 0.55, calibrated empirically by running sample queries. On its own it is not a sufficient off-topic gate — see Reranking.

### 4. Reranking

- **Model:** `bge-reranker-base` cross-encoder, set up from scratch via `transformers`.
- **Flow:** Bi-encoder retrieves top-10 candidates → cross-encoder rescores and sorts descending → chunks scoring above 0 pass to the LLM, capped at 5.
- **Relevance gate:** The threshold of 0 was calibrated on a 12-query sweep — on-topic chunks scored 0.57 to 7.19, off-topic scored −6.26 to −1.87. A query with nothing above 0 gets a "no relevant material" reply and never reaches the LLM.
- **Hardware:** MPS, same as the bi-encoder.

### 5. LLM Integration

- **Model:** `openai/gpt-oss-120b` via Groq API (128K context window).
- **Prompting:** Each retrieved chunk is passed as a structured block — `chunk N | chunk_id | source | text` — so the model can cite which context it drew from, with grounding rules to prevent answering outside the retrieved material. The reply must end with a `CITATIONS: <id>, <id>` line, which the output guardrail parses; inline `chunk_id` mentions are accepted as a fallback, since the model's inline format varies between runs.

### 6. Guardrails

- **Input:** Empty and over-length queries are rejected, alongside regex screens for prompt injection, off-syllabus misuse, and offensive language. Patterns are anchored to persona-switching phrasing rather than bare keywords, so subject vocabulary survives — "acids act as proton donors" is a chemistry question, not a jailbreak. Current test set: 13/13 attacks blocked, 0/10 legitimate JEE queries blocked.
- **Output:** The chunk_ids the model cites are intersected with the ids it was given. No overlap means the answer is served with a hallucination warning appended.

### 7. Resilience & Observability

Custom retry/fallback wrapper around the Groq client — `max_retries=0` and `timeout=30s` on the SDK so the custom loop owns retry behavior, with exponential backoff across 3 attempts.

Errors are split by whether a retry can help. `BadRequestError`, `AuthenticationError` and `NotFoundError` return immediately with the real reason logged; rate limits, 5xx, timeouts and connection failures are retried. Each stage logs its latency and every call logs token usage.

Still planned: structured logging and an explicit token budget.

---

## Decision Log: Rejected Approaches

Part of the point of this rebuild is being able to defend every component. The full unedited trial-and-error history lives in [`Decisions.md`](Decisions.md); the summary of what was tried and killed:

**Extraction — all trials run against the NCERT corpus:**

| Approach | Verdict | Reason |
|---|---|---|
| **pypdf** | ❌ Rejected | Garbled Unicode (custom CMap font encoding) |
| **PyMuPDF** | ❌ Same failure | Identical garbled output — pointed to the PDFs, not the library |
| **Docling + OCR** | ❌ Rejected | Prohibitively slow; MPS float64 incompatibility forced CPU-only |
| **Docling without OCR** | ❌ Rejected | Table-layout-aware mode produced worse output than baseline |
| **Marker-pdf** | ❌ Rejected | Too RAM-hungry for the target machine (8GB M1) |
| **PIL + pytesseract** | ❌ Rejected | Failed to extract question text across multiple OEM/PSM configs |
| **Docling OCR on Colab** | ❌ Rejected | Still failed |

With every extraction path dead against NCERT, the diagnosis shifted from *tooling* to *corpus*: the font-cipher corruption meant no extractor could win. **NCERT was replaced with OpenStax**, and PyMuPDF — which had already failed on NCERT — worked cleanly on the new corpus, confirming the diagnosis.

**Other rejected approaches:**

| Approach | Verdict | Reason |
|---|---|---|
| **Manual character splitter** | ❌ Rejected | No overlap, no metadata, degenerate chunk sizes (1–440 chars) |
| **FAISS / vector DB** | ❌ Not adopted | 5,202 chunks × 768 dims is a single dot product per query, sub-millisecond in NumPy. An index adds a dependency and a build step for no measurable gain at this corpus size. |

**Final stack:** PyMuPDF + custom filtering on OpenStax → LangChain recursive splitting → BGE + NumPy vector search → bge-reranker-base → Groq.

---

## Roadmap

- [ ] Retrieval evaluation: MRR and related metrics on a labeled query set
- [ ] Compare asymmetric vs. symmetric BGE variants with metrics
- [ ] Prompt evaluation: compare answer quality across prompt structures against an eval dataset
- [ ] Semantic chunking experiment (post-retrieval-baseline)
- [x] Resilience layer: retry with backoff, timeouts, non-retryable error handling
- [ ] Token limits and structured logging — usage is measured per call (~2,300–2,500 prompt, ~1,000–1,600 completion) but not yet budgeted
- [x] Backend: FastAPI + minimal chat frontend
- [ ] Rate limiting, and a health check that verifies the index is loaded
- [ ] Deployment
- [ ] Router: maths and calculation via an external library, physics/chemistry theory, syllabus lookup

---

## Project Structure

```
AI-JEE-Agent/
├── Extraction.py        # PDF → cleaned .txt (PyMuPDF + junk-line filtering)
├── Chunking.py          # Token-aware recursive splitting + metadata
├── embeddings.py        # BGE encoding → embeddings.npy + chunks.json
├── reranker.py          # Cross-encoder rescoring + relevance gate
├── guardrails.py        # Input screening + output citation check
├── integration.py       # Retrieval → rerank → prompt → Groq LLM
├── app.py               # FastAPI backend (/, /health, /chat)
├── index.html           # Minimal chat frontend
├── Decisions.md         # Raw decision log (full trial-and-error history)
├── state.md             # Running notes: open bugs, latency measurements
├── data/                # Source PDFs (OpenStax)
├── chunks/              # Chunked corpus
├── embeddings/
│   ├── chunks.json      # Chunk texts + metadata (parallel to embeddings)
│   └── embeddings.npy   # Precomputed BGE vectors
├── output/              # Extracted plain text, one .txt per source PDF
└── images/              # Documentation assets
```

## Setup

```bash
git clone https://github.com/Sparsh-25/AI-JEE-AGENT-v2.git
cd AI-JEE-Agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
API_KEY=your-key-here
```

Run the pipeline in order (each stage writes artifacts the next stage reads):

```bash
python Extraction.py     # data/ PDFs → cleaned text
python Chunking.py       # text → chunks/
python embeddings.py     # chunks → embeddings/embeddings.npy + chunks.json
python integration.py    # ask a question from the CLI
```

Then start the server:

```bash
uvicorn app:app --reload
```

The chat frontend is served at `http://127.0.0.1:8000/`.

---

## Tech Stack

**Extraction:** PyMuPDF · **Chunking:** LangChain, HuggingFace tokenizers · **Embeddings:** BGE (bi-encoder) · **Retrieval:** NumPy dot product · **Reranking:** bge-reranker-base (cross-encoder) · **LLM:** Groq (openai/gpt-oss-120b) · **Serving:** FastAPI, Uvicorn (Docker and Hugging Face Spaces deployment still to do)
