"""
classifier/models.py — Pydantic models for classifier input/output.

Strict validation at every system boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from config.routes import DocumentType


# ══════════════════════════════════════════════════════════════════════
#  Classification Output
# ══════════════════════════════════════════════════════════════════════


class ClassificationCandidate(BaseModel):
    """A candidate in the classification top-k."""

    document_type: DocumentType
    confidence: float = Field(..., ge=-1.0, le=1.0)

    model_config = ConfigDict(frozen=True)


class ClassificationResult(BaseModel):
    """Complete result of a semantic classification."""

    model_config = ConfigDict(frozen=True)

    document_type: DocumentType = Field(
        ..., description="Predicted type (INCONNU if below threshold)"
    )
    confidence: float = Field(
        ..., ge=-1.0, le=1.0, description="Cosine similarity of the predicted type"
    )
    top_k: list[ClassificationCandidate] = Field(
        default_factory=list,
        description="Top-k candidate classes with scores",
    )
    embedding_model: str = Field(..., description="ID of the model used")
    confidence_threshold: float = Field(
        ..., description="Threshold applied for this result"
    )
    latency_ms: float = Field(..., ge=0, description="Inference time in ms")

    @property
    def is_confident(self) -> bool:
        """True if the classification is above the threshold."""
        return self.document_type != DocumentType.INCONNU

    @property
    def margin(self) -> float:
        """Margin between the 1st and 2nd candidate. Higher means more certain."""
        if len(self.top_k) < 2:
            return self.confidence
        return self.top_k[0].confidence - self.top_k[1].confidence


# ══════════════════════════════════════════════════════════════════════
#  Pipeline I/O
# ══════════════════════════════════════════════════════════════════════


class DocumentInput(BaseModel):
    """Input document for the pipeline."""

    model_config = ConfigDict(strict=True)

    file_path: Path = Field(..., description="Path to the source file")
    document_id: str = Field(
        default="",
        description="Unique ID. Automatically generated if empty.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata attached to the document",
    )

    @field_validator("file_path")
    @classmethod
    def file_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"File does not exist: {v}")
        return v

    @model_validator(mode="after")
    def generate_id_if_empty(self) -> "DocumentInput":
        if not self.document_id:
            import uuid
            object.__setattr__(self, "document_id", str(uuid.uuid4()))
        return self


class PipelineResult(BaseModel):
    """Complete pipeline result for a document."""

    document_id: str
    file_path: str
    classification: ClassificationResult
    ocr_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Average OCR confidence"
    )
    text_length: int = Field(..., ge=0, description="Length of the extracted text")
    text_preview: str = Field(
        default="", max_length=500, description="Preview of the extracted text"
    )
    total_latency_ms: float = Field(..., ge=0)
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    warnings: list[str] = Field(default_factory=list)


class BatchResult(BaseModel):
    """Result of a batch processing run."""

    total_documents: int
    successful: int
    failed: int
    results: list[PipelineResult]
    errors: list[dict[str, str]] = Field(default_factory=list)
    total_latency_ms: float
    throughput_docs_per_sec: float
