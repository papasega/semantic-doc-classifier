"""
main.py — FastAPI application for document classification.

Endpoints:
    POST /classify          -> Single file classification
    POST /classify/batch    -> Batch classification
    POST /classify/explain  -> Detailed explanation (debug)
    GET  /health            -> Health check
    GET  /routes            -> Configured semantic routes
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import get_settings
from classifier.pipeline import ClassificationPipeline
from classifier.models import PipelineResult, BatchResult
from classifier.semantic_router import SemanticRouter

logger = logging.getLogger(__name__)

# ── Globals (initialized at startup) ─────────────────────────────
_pipeline: ClassificationPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models ONCE at server startup."""
    global _pipeline
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    logger.info("Initializing classification pipeline...")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    _pipeline = ClassificationPipeline()

    logger.info(
        "Pipeline ready — %d routes, model: %s, device: %s",
        _pipeline._router.n_routes,
        _pipeline._router._engine.model_id,
        _pipeline._router._engine.device,
    )
    yield
    logger.info("Pipeline shutdown")


app = FastAPI(
    title="Document Intelligence — Classification API",
    version="1.0.0",
    description=(
        "Semantic document classification via E5 embeddings and vector routing. "
        "Zero if/else, zero regex. Pure vector geometry."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════
#  Endpoints
# ══════════════════════════════════════════════════════════════════════


@app.post("/classify", response_model=PipelineResult)
async def classify_document(file: UploadFile = File(...)):
    """Classify a single document.

    Accepts: PDF, DOCX, TXT, PNG, JPG, TIFF, BMP, WebP.

    Returns:
        PipelineResult with classification, OCR confidence and metrics.
    """
    settings = get_settings()
    doc_id = str(uuid.uuid4())
    save_path = settings.upload_dir / f"{doc_id}_{file.filename}"

    try:
        async with aiofiles.open(save_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        result = _pipeline.classify_file(save_path)
        return result

    except Exception as e:
        logger.exception("Classification error for %s", file.filename)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Temporary file cleanup
        if save_path.exists():
            save_path.unlink(missing_ok=True)


@app.post("/classify/batch", response_model=BatchResult)
async def classify_batch(files: list[UploadFile] = File(...)):
    """Classify a batch of documents.

    Phase 1 (sequential): Upload + text extraction.
    Phase 2 (GPU batch):  Vector classification.

    Returns:
        BatchResult with individual results and global metrics.
    """
    settings = get_settings()
    saved_paths: list[Path] = []

    try:
        for file in files:
            doc_id = str(uuid.uuid4())
            save_path = settings.upload_dir / f"{doc_id}_{file.filename}"
            async with aiofiles.open(save_path, "wb") as f:
                await f.write(await file.read())
            saved_paths.append(save_path)

        result = _pipeline.classify_batch(saved_paths)
        return result

    except Exception as e:
        logger.exception("Batch classification error")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for p in saved_paths:
            if p.exists():
                p.unlink(missing_ok=True)


class ExplainRequest(BaseModel):
    text: str


@app.post("/classify/explain")
async def explain_classification(request: ExplainRequest) -> dict[str, Any]:
    """Detailed classification explanation.

    Returns scores for ALL classes, the margin between the 1st and 2nd
    candidate, and a preview of the analyzed text.

    Usage: debug, human validation, threshold tuning.
    """
    return _pipeline._router.explain(request.text)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check with model details."""
    return {
        "status": "healthy",
        "model": _pipeline._router._engine.model_id,
        "device": _pipeline._router._engine.device,
        "n_routes": _pipeline._router.n_routes,
        "route_types": [rt.value for rt in _pipeline._router.route_types],
        "confidence_threshold": _pipeline._router.confidence_threshold,
        "embedding_dim": _pipeline._router.embedding_dim,
    }


@app.get("/routes")
async def list_routes() -> dict[str, Any]:
    """List configured semantic routes.

    Useful for checking which classes are active
    and which descriptions define them.
    """
    from config.routes import get_routes

    routes = get_routes()
    return {
        "n_routes": len(routes),
        "routes": [
            {
                "type": r.document_type.value,
                "n_descriptions": len(r.descriptions),
                "descriptions": list(r.descriptions),
            }
            for r in routes
        ],
    }
