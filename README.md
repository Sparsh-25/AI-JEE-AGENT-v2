# AI JEE Agent v2

> 🚧 **Work in progress.** This is a ground-up rebuild of the [AI JEE Agent](https://huggingface.co/spaces/) — the live demo above is **v1**. v2 is being written by hand, component by component, and is not yet complete. Current state: extraction → chunking → embeddings → reranking → LLM integration are working; evaluation and the resilience layer are in progress (see [Roadmap](#roadmap)).

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
Vector Retrieval (FAISS, cosine similarity, threshold = 0.6)
   │
   ▼
Reranking (bge-reranker-base cross-encoder, top-20 → ranked)
   │
   ▼
Prompt Construction (structured chunk blocks + grounding rules)
   │
   ▼
LLM (Groq · llama-3.3-70b-versatile · 128K context)
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
- **Retrieval threshold:** Cosine similarity 0.6, calibrated empirically by running sample queries.

### 4. Reranking

- **Model:** `bge-reranker-base` cross-encoder, set up from scratch via `transformers`.
- **Flow:** Bi-encoder retrieves top-20 candidates → cross-encoder rescores and sorts descending → passed to the LLM.

### 5. LLM Integration

- **Model:** `llama-3.3-70b-versatile` via Groq API (128K context window).
- **Prompting:** Each retrieved chunk is passed as a structured block — `chunk N | chunk_id | source | text` — so the model can cite which context it drew from, with grounding rules to prevent answering outside the retrieved material.

### 6. Resilience & Observability *(in progress)*

Custom retry/fallback wrapper around the Groq client (`max_retries=0` on the SDK so custom logic owns retry behavior), with planned token counting, timeout handling, and logging.

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

**Final stack:** PyMuPDF + custom filtering on OpenStax → LangChain recursive splitting → BGE + FAISS → bge-reranker-base → Groq.

---

## Roadmap

- [ ] Retrieval evaluation: MRR and related metrics on a labeled query set
- [ ] Compare asymmetric vs. symmetric BGE variants with metrics
- [ ] Cap reranked context at top-5 chunks (currently passing all 20)
- [ ] Prompt evaluation: compare answer quality across prompt structures against an eval dataset
- [ ] Measure total prompt token length (input + output share the context window) and budget accordingly
- [ ] Semantic chunking experiment (post-retrieval-baseline)
- [ ] Full resilience layer: token limits, logging, retry with backoff, timeouts

---

## Project Structure

```
AI-JEE-Agent/
├── Extraction.py        # PDF → cleaned .txt (PyMuPDF + junk-line filtering)
├── Chunking.py          # Token-aware recursive splitting + metadata
├── embeddings.py        # BGE encoding → embeddings.npy + chunks.json
├── reranker.py          # Cross-encoder rescoring (bge-reranker-base)
├── integration.py       # Retrieval → rerank → prompt → Groq LLM
├── Decisions.md         # Raw decision log (full trial-and-error history)
├── data/                # Source PDFs (OpenStax)
├── chunks/              # Chunked corpus
├── embeddings/
│   ├── chunks.json      # Chunk texts + metadata (parallel to embeddings)
│   └── embeddings.npy   # Precomputed BGE vectors
├── output/              # Generated answers / run artifacts
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
GROQ_API_KEY=your-key-here
```

Run the pipeline in order (each stage writes artifacts the next stage reads):

```bash
python Extraction.py     # data/ PDFs → cleaned text
python Chunking.py       # text → chunks/
python embeddings.py     # chunks → embeddings/embeddings.npy + chunks.json
python integration.py    # ask questions: retrieve → rerank → answer
```

---

## Tech Stack

**Extraction:** PyMuPDF · **Chunking:** LangChain, HuggingFace tokenizers · **Embeddings:** BGE (bi-encoder) · **Retrieval:** FAISS · **Reranking:** bge-reranker-base (cross-encoder) · **LLM:** Groq (llama-3.3-70b-versatile) · **Serving:** FastAPI, Docker, Hugging Face Spaces
