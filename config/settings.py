"""
config/settings.py — Centralized pipeline configuration.

Uses pydantic-settings for:
- Type validation at startup
- .env / environment variable support
- Sane defaults for dev, overridable in prod
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class PipelineSettings(BaseSettings):
    """Single configuration object for the entire pipeline."""

    model_config = {"env_prefix": "DIP_", "env_file": ".env", "extra": "ignore"}

    # ── Embedding Engine ──────────────────────────────────────────────
    embedding_model_id: str = Field(
        default="intfloat/multilingual-e5-large",
        description="HuggingFace model ID for embeddings",
    )
    embedding_device: str = Field(
        default="auto",
        description="Device: 'cuda', 'cpu', or 'auto' (automatic detection)",
    )
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Batch size for GPU encoding. Adjust based on available VRAM.",
    )
    embedding_max_seq_length: int = Field(
        default=512,
        ge=64,
        le=8192,
        description="Max sequence length for the tokenizer",
    )

    # ── OCR Engine ────────────────────────────────────────────────────
    ocr_lang: str = Field(default="fr", description="PaddleOCR language")
    ocr_use_gpu: bool = Field(default=True)
    ocr_use_angle_cls: bool = Field(
        default=True,
        description="Orientation detection — CRUCIAL for scanned documents",
    )
    ocr_det_db_box_thresh: float = Field(default=0.5, ge=0.1, le=0.9)
    ocr_rec_batch_num: int = Field(
        default=6,
        ge=1,
        le=32,
        description="OCR recognition batch size. Adjust based on VRAM.",
    )

    # ── Semantic Router ───────────────────────────────────────────────
    confidence_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence threshold below which a document is classified as UNKNOWN. "
            "0.40 is empirically a good default for E5-large with 5-8 classes."
        ),
    )
    classification_top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of candidate classes returned",
    )
    text_truncation_chars: int = Field(
        default=4096,
        ge=256,
        le=32768,
        description="Text truncation before embedding. E5 supports 512 tokens.",
    )

    # ── Pipeline ──────────────────────────────────────────────────────
    upload_dir: Path = Field(
        default=Path("/data/uploads"),
        description="Storage directory for uploaded documents",
    )
    supported_image_extensions: frozenset[str] = Field(
        default=frozenset({".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}),
    )
    supported_doc_extensions: frozenset[str] = Field(
        default=frozenset({".pdf", ".docx", ".txt"}),
    )
    max_file_size_mb: int = Field(default=100, ge=1, le=500)

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    @field_validator("embedding_device", mode="before")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        if v == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v


@lru_cache(maxsize=1)
def get_settings() -> PipelineSettings:
    """Thread-safe settings singleton."""
    return PipelineSettings()
