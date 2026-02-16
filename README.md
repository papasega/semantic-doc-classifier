# semantic-doc-classifier

**Production-grade document classification using vector embeddings and cosine similarity.**
**Zero regex. Zero keyword matching. Pure vector geometry.**

> For the full mathematical theory behind embeddings, OCR, and classification, see **[THEORY.md](THEORY.md)**.

---

## How It Works

```
Document → Text Extraction → Embedding (E5-large, 1024-dim) → Cosine Similarity vs Centroids → Class
              │                        │                                  │
         PaddleOCR /              L2-normalized                    argmax + threshold
         PyMuPDF / docx           "query:" prefix                  → FACTURE | CONTRAT | ...
```

Each document class is defined by **natural language descriptions** (not keywords). At startup, these descriptions are embedded into vectors and averaged into a **centroid** per class. At inference, the document text is embedded and compared against all centroids via dot product. The highest-scoring class wins.

```python
# The entire classification logic — no rules, no regex
similarities = query_embedding @ centroids.T
predicted_class = argmax(similarities)
```

**Why this beats rule-based approaches:**

- "note de débit" is correctly classified as an invoice — semantically close, no keyword needed
- Works across French, English, and mixed-language documents out of the box
- OCR typos ("factur3") don't break classification — the embedding captures meaning, not spelling
- Adding a new class = adding 4-6 descriptions in one config file, zero code changes

---

## Project Structure

```
doc_classifier/
├── config/
│   ├── settings.py              # Pydantic Settings (env vars, defaults)
│   └── routes.py                # ⭐ Semantic route definitions (add classes here)
├── engines/
│   ├── ocr_engine.py            # PaddleOCR wrapper (hardened)
│   └── embedding_engine.py      # SentenceTransformer E5 wrapper
├── classifier/
│   ├── models.py                # Pydantic I/O models
│   ├── semantic_router.py       # Core: centroid-based cosine classification
│   └── pipeline.py              # Orchestrator: File → Text → Embed → Classify
├── tests/
│   └── test_router.py           # 7 classes × realistic FR texts + edge cases
├── main.py                      # FastAPI API
├── Dockerfile
├── docker-compose.yml
├── THEORY.md                    # Mathematical deep-dive
└── requirements.txt
```

---

## Quick Start

### Docker (recommended)

```bash
docker compose up -d
curl http://localhost:8000/health

# Classify a document
curl -X POST http://localhost:8000/classify -F "file=@invoice.pdf"

# Batch
curl -X POST http://localhost:8000/classify/batch \
  -F "files=@doc1.pdf" -F "files=@doc2.pdf"
```

### Local

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Python API

```python
from pathlib import Path
from classifier.pipeline import ClassificationPipeline

pipeline = ClassificationPipeline()

# Single file
result = pipeline.classify_file(Path("scan_contrat.pdf"))
print(result.classification.document_type)   # DocumentType.CONTRAT
print(result.classification.confidence)      # 0.847
print(result.classification.margin)          # 0.213

# Batch (GPU-optimized — single matrix multiply)
batch = pipeline.classify_batch([Path("f1.pdf"), Path("f2.pdf"), Path("f3.docx")])
print(f"{batch.throughput_docs_per_sec:.0f} docs/sec")
```

### Debug / Explain

```python
from classifier.semantic_router import SemanticRouter

router = SemanticRouter()
print(router.explain("FACTURE N° 2024 — Total TTC: 15 300€"))
# {
#   "predicted": "facture",
#   "confidence": 0.847,
#   "margin": 0.213,
#   "all_scores": {"facture": 0.847, "contrat": 0.634, "rapport": 0.592, ...}
# }
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/classify` | POST | Classify single file (multipart upload) |
| `/classify/batch` | POST | Classify multiple files (GPU batch) |
| `/classify/explain` | POST | Detailed score breakdown (JSON: `{"text": "..."}`) |
| `/health` | GET | Model info, device, routes |
| `/routes` | GET | All configured semantic routes |

---

## Configuration

All settings via **environment variables** (prefix `DIP_`) or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `DIP_EMBEDDING_MODEL_ID` | `intfloat/multilingual-e5-large` | HuggingFace model |
| `DIP_EMBEDDING_DEVICE` | `auto` | `cuda` / `cpu` / `auto` |
| `DIP_EMBEDDING_BATCH_SIZE` | `32` | GPU batch size (adjust to VRAM) |
| `DIP_OCR_LANG` | `fr` | PaddleOCR language |
| `DIP_OCR_USE_GPU` | `true` | GPU for OCR |
| `DIP_CONFIDENCE_THRESHOLD` | `0.40` | Below this → classified as UNKNOWN |
| `DIP_CLASSIFICATION_TOP_K` | `3` | Candidate classes returned |

---

## Adding a New Document Class

Edit **only** `config/routes.py`:

```python
# 1. Add to the enum
class DocumentType(str, Enum):
    # ... existing ...
    DEVIS = "devis"

# 2. Add a RoutePrototype
ROUTES.append(
    RoutePrototype(
        document_type=DocumentType.DEVIS,
        descriptions=(
            "Proposition commerciale chiffrée avec détail des prestations et prix",
            "Cost estimate or quotation with line items and validity period",
            "Devis détaillé avec conditions de vente et délai de validité",
            "Commercial offer listing services, unit prices and total cost",
        ),
    )
)
```

**That's it.** No code changes elsewhere. The centroid is computed automatically at startup.

> **Tips for good descriptions:** Describe the *concept*, not keywords. Mix FR + EN. Use 4–8 descriptions per class. Include variations (formal/informal, synonyms, related sub-types).

---

## Supported File Formats

| Format | Extraction Method | Notes |
|--------|-------------------|-------|
| PDF (digital) | PyMuPDF native text | Fast, high quality |
| PDF (scanned) | PyMuPDF rasterize → PaddleOCR | Auto-detected per page |
| DOCX | python-docx | Native text extraction |
| Images (PNG, JPG, TIFF, BMP, WebP) | PaddleOCR | With angle correction |
| TXT | Direct read | UTF-8 with latin-1 fallback |

---

## Performance

| Metric | T4 (16GB) | A10G (24GB) | A100 (40GB) |
|--------|-----------|-------------|-------------|
| Model load | ~8s | ~5s | ~3s |
| Single classify | ~50ms | ~35ms | ~25ms |
| Batch (100 docs) | ~500ms | ~300ms | ~200ms |
| Throughput | ~200/s | ~350/s | ~500/s |
| VRAM | ~4.5GB | ~4.5GB | ~4.5GB |

OCR adds 100–500ms per page depending on complexity and DPI.

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=classifier --cov-report=term-missing
```

Covers: 7 document types with realistic French text, rejection cases (lorem ipsum, gibberish), batch/single consistency, centroid normalization, edge cases (empty text, very long text).

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Embedding | `intfloat/multilingual-e5-large` | Top MTEB for FR/EN, 1024-dim, instruction-tuned |
| OCR | PaddleOCR | Fast, multilingual, angle detection, production-proven |
| Validation | Pydantic v2 | Strict types, serialization, settings management |
| API | FastAPI | Async, OpenAPI docs, lightweight |
| PDF parsing | PyMuPDF | Native text + rasterization for OCR fallback |

---

## What's Next (Roadmap)

- **Step 2:** Vector storage (Qdrant + HNSW indexing) for document retrieval
- **Step 3:** RAG pipeline with semantic chunking + cross-encoder reranking
- **Step 4:** Async workers (Celery + Redis) for 10,000+ document ingestion

---

*Orange DATA-IA / SND / DREAMS*
