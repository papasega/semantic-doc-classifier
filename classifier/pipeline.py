"""
classifier/pipeline.py — Classification pipeline orchestrator.

Full flow:
    File -> Text extraction (OCR/PDF/DOCX) -> SemanticRouter -> Result

Handles:
- Automatic format detection (native PDF vs scan, DOCX, images)
- PDF text extraction with per-page OCR fallback
- Multi-page result aggregation
- End-to-end metrics
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from config.settings import get_settings
from engines.ocr_engine import OCREngine, OCRResult
from engines.embedding_engine import get_embedding_engine
from classifier.semantic_router import SemanticRouter
from classifier.models import (
    DocumentInput,
    PipelineResult,
    BatchResult,
)

logger = logging.getLogger(__name__)


class ClassificationPipeline:
    """Full pipeline: File -> Text -> Classification.

    Responsibilities:
    1. Detect file type and extract text
    2. Delegate classification to SemanticRouter
    3. Assemble the PipelineResult with metrics

    Usage:
        pipeline = ClassificationPipeline()
        result = pipeline.classify_file(Path("scan_contrat.pdf"))
        print(result.classification.document_type)  # "contrat"
    """

    def __init__(
        self,
        ocr_engine: OCREngine | None = None,
        router: SemanticRouter | None = None,
    ) -> None:
        settings = get_settings()
        self._settings = settings
        self._ocr = ocr_engine or OCREngine()
        self._router = router or SemanticRouter()

    # ── Public API ────────────────────────────────────────────────

    def classify_file(self, file_path: Path | str) -> PipelineResult:
        """Classify a single file.

        Supports: PDF (native + scanned), DOCX, TXT, images (PNG/JPG/TIFF/BMP).

        Args:
            file_path: Path to the file.

        Returns:
            Complete PipelineResult with classification and metrics.
        """
        t0 = time.perf_counter()
        path = Path(file_path)
        doc_input = DocumentInput(file_path=path)
        warnings: list[str] = []

        # ── Text extraction ────────────────────────────────────────
        text, ocr_confidence = self._extract_text(path, warnings)

        if not text.strip():
            warnings.append("No text extracted from the document")
            # Still classify: the router will return INCONNU
            # via the confidence threshold, not via an if/else here.

        # ── Semantic classification ────────────────────────────────
        classification = self._router.classify(text)

        total_ms = (time.perf_counter() - t0) * 1000

        return PipelineResult(
            document_id=doc_input.document_id,
            file_path=str(path),
            classification=classification,
            ocr_confidence=ocr_confidence,
            text_length=len(text),
            text_preview=text[:500],
            total_latency_ms=round(total_ms, 2),
            warnings=warnings,
        )

    def classify_batch(
        self, file_paths: list[Path | str]
    ) -> BatchResult:
        """Classify a batch of files.

        Steps:
        1. Sequential text extraction (I/O bound -> no GPU)
        2. GPU batch classification (single call)

        Args:
            file_paths: List of file paths.

        Returns:
            BatchResult with results, errors and metrics.
        """
        t0 = time.perf_counter()
        results: list[PipelineResult] = []
        errors: list[dict[str, str]] = []
        texts: list[str] = []
        extraction_data: list[dict] = []

        # ── Phase 1: Text extraction (sequential) ─────────────────
        for fp in file_paths:
            path = Path(fp)
            try:
                doc_input = DocumentInput(file_path=path)
                warnings: list[str] = []
                text, ocr_conf = self._extract_text(path, warnings)
                texts.append(text)
                extraction_data.append({
                    "doc_input": doc_input,
                    "path": path,
                    "ocr_confidence": ocr_conf,
                    "text_length": len(text),
                    "text_preview": text[:500],
                    "warnings": warnings,
                })
            except Exception as e:
                logger.error("Extraction failed for %s: %s", fp, e)
                errors.append({"file": str(fp), "error": str(e)})

        # ── Phase 2: GPU batch classification ─────────────────────
        classifications = []
        if texts:
            classifications = self._router.classify_batch(texts)

        # ── Phase 3: Result assembly ──────────────────────────────
        total_ms = (time.perf_counter() - t0) * 1000

        for i, cls_result in enumerate(classifications):
            data = extraction_data[i]
            results.append(PipelineResult(
                document_id=data["doc_input"].document_id,
                file_path=str(data["path"]),
                classification=cls_result,
                ocr_confidence=data["ocr_confidence"],
                text_length=data["text_length"],
                text_preview=data["text_preview"],
                total_latency_ms=round(total_ms / len(texts), 2),
                warnings=data["warnings"],
            ))

        throughput = len(results) / (total_ms / 1000) if total_ms > 0 else 0

        return BatchResult(
            total_documents=len(file_paths),
            successful=len(results),
            failed=len(errors),
            results=results,
            errors=errors,
            total_latency_ms=round(total_ms, 2),
            throughput_docs_per_sec=round(throughput, 1),
        )

    # ── Text Extraction ───────────────────────────────────────────

    def _extract_text(
        self, path: Path, warnings: list[str]
    ) -> tuple[str, float]:
        """Extract text from a file based on its type.

        Returns:
            Tuple (text, ocr_confidence).
            ocr_confidence = 1.0 for native text files.
        """
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._extract_from_pdf(path, warnings)
        elif suffix == ".docx":
            return self._extract_from_docx(path, warnings)
        elif suffix == ".txt":
            return self._extract_from_txt(path, warnings)
        elif suffix in self._settings.supported_image_extensions:
            return self._extract_from_image(path, warnings)
        else:
            warnings.append(f"Unrecognized extension: {suffix}, attempting OCR")
            return self._extract_from_image(path, warnings)

    def _extract_from_pdf(
        self, path: Path, warnings: list[str]
    ) -> tuple[str, float]:
        """Extract text from a PDF.

        Dual strategy:
        1. Attempt native text extraction (PyMuPDF)
        2. If a page has little text -> OCR fallback on that page
        """
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        all_text_parts: list[str] = []
        ocr_confidences: list[float] = []
        pages_ocr_count = 0

        for page_num, page in enumerate(doc):
            # Native text extraction
            native_text = page.get_text().strip()

            if len(native_text) > 50:
                # Page with native text -> no OCR needed
                all_text_parts.append(native_text)
                ocr_confidences.append(1.0)
            else:
                # Scanned or image page -> OCR
                pages_ocr_count += 1
                pix = page.get_pixmap(dpi=300)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                # Convert to BGR for OpenCV/PaddleOCR
                if pix.n == 4:
                    import cv2
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    import cv2
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                ocr_result = self._ocr.process_image_array(
                    img_array,
                    source_label=f"{path.name}:page{page_num}",
                    page_index=page_num,
                )
                all_text_parts.append(ocr_result.raw_text)
                ocr_confidences.append(ocr_result.average_confidence)

                if ocr_result.is_low_quality:
                    warnings.append(
                        f"Page {page_num}: low quality OCR "
                        f"(confidence={ocr_result.average_confidence:.2f})"
                    )

        doc.close()

        if pages_ocr_count > 0:
            logger.info(
                "PDF %s: %d native text pages, %d OCR pages",
                path.name,
                len(all_text_parts) - pages_ocr_count,
                pages_ocr_count,
            )

        full_text = "\n\n".join(all_text_parts)
        avg_confidence = float(np.mean(ocr_confidences)) if ocr_confidences else 0.0

        return full_text, avg_confidence

    def _extract_from_docx(
        self, path: Path, warnings: list[str]
    ) -> tuple[str, float]:
        """Extract text from a Word (.docx) file."""
        from docx import Document

        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        if not paragraphs:
            warnings.append("Empty DOCX document or no text paragraphs")

        return "\n".join(paragraphs), 1.0  # Native text = confidence 1.0

    def _extract_from_txt(
        self, path: Path, warnings: list[str]
    ) -> tuple[str, float]:
        """Extract text from a plain text file."""
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
            warnings.append("File decoded as latin-1 (not UTF-8)")
        return text, 1.0

    def _extract_from_image(
        self, path: Path, warnings: list[str]
    ) -> tuple[str, float]:
        """Extract text from an image via OCR."""
        ocr_result = self._ocr.process_image(str(path))

        if ocr_result.is_empty:
            warnings.append(f"OCR detected no text in {path.name}")
        elif ocr_result.is_low_quality:
            warnings.append(
                f"Low quality OCR on {path.name} "
                f"(confidence={ocr_result.average_confidence:.2f})"
            )

        return ocr_result.raw_text, ocr_result.average_confidence
