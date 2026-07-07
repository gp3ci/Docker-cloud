"""
Stateless Service for generating the 'Before' map overview.
Adds Top-Right Survey Info block without invoking ML models.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import fitz

from app.core.config import Settings
from app.models.schemas import JobStatus
from app.services.alignment import pdf_to_image
from app.services.reporting import _draw_legend_stack

logger = logging.getLogger(__name__)


def run_fiber_before_pipeline(
    job_id: str,
    job_store: dict,
    settings: Settings,
) -> None:
    """
    Stateless processing for the Before map. Only converts the PDF and stamps
    the Survey Image & Title Box block using the predefined robust styling.
    """
    job_start = time.perf_counter()

    def _update(status: JobStatus, pct: float, msg: str) -> None:
        job_store[job_id].update({"status": status, "progress": pct, "message": msg})
        logger.info(f"[{job_id}] [{pct:3.0f}%] {msg}")

    def _record(stage: str, t0: float) -> float:
        elapsed = (time.perf_counter() - t0) * 1000
        job_store[job_id]["stage_times"][stage] = round(elapsed, 1)
        return time.perf_counter()

    try:
        job = job_store[job_id]
        pdf_path = Path(job["pdf_path"])
        output_dir = Path(job["output_dir"])
        dpi = job.get("dpi", settings.PDF_DPI)
        survey_image_path = job.get("survey_image_path")
        title_box_data = job.get("title_box", {})

        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        job_store[job_id]["stage_times"] = {}
        t0 = time.perf_counter()

        # ── Download from GCS if running on RunPod (file won't exist locally) ──
        if not pdf_path.exists():
            gcs_key = job.get("pdf_path_gcs") or job.get("before_path_gcs")
            if gcs_key:
                try:
                    from app.services.storage import download_from_storage
                    _update(JobStatus.PROCESSING, 5, "Downloading PDF from cloud storage...")
                    download_from_storage(gcs_key, pdf_path)
                    logger.info(f"[{job_id}] Downloaded PDF from GCS: {gcs_key}")
                except Exception as dl_err:
                    raise FileNotFoundError(f"PDF not found locally and GCS download failed: {dl_err}")
            else:
                raise FileNotFoundError(f"PDF not found at {pdf_path} and no GCS path available.")

        # ── Download survey image from GCS if needed ──
        if survey_image_path:
            survey_path_obj = Path(survey_image_path)
            if not survey_path_obj.exists():
                survey_gcs = job.get("survey_image_path_gcs")
                if survey_gcs:
                    try:
                        from app.services.storage import download_from_storage
                        survey_path_obj.parent.mkdir(parents=True, exist_ok=True)
                        download_from_storage(survey_gcs, survey_path_obj)
                    except Exception:
                        survey_image_path = None  # proceed without it

        _update(JobStatus.PROCESSING, 10, "Extracting PDF raster bounds...")
        img = pdf_to_image(pdf_path, dpi=dpi)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t0 = _record("CONVERSION", t0)

        _update(JobStatus.REPORTING, 50, "Stamping native Survey and Title Box overlays...")
        report_path = output_dir / "report.pdf"
        
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        
        # Stamp title box on EVERY page, survey image only on page 0
        for pg_idx in range(total_pages):
            page = doc.load_page(pg_idx)
            page_title_box = dict(title_box_data) if title_box_data else {}
            page_title_box["page_count"] = total_pages
            
            _draw_legend_stack(
                page=page,
                img_gray=img_gray,
                callouts=[],
                survey_image_path=survey_image_path if pg_idx == 0 else None,
                title_box_data=page_title_box,
                dpi=dpi,
                include_legend=False,
                title_font_size=34,
                page_num=pg_idx + 1,
                total_pages=total_pages,
            )
        
        doc.save(str(report_path), deflate=True, garbage=4, clean=True, linear=False)
        doc.close()
        t0 = _record("REPORTING", t0)

        total_ms = (time.perf_counter() - job_start) * 1000
        job_store[job_id]["stage_times"]["total_ms"] = round(total_ms, 1)

        # ── Upload report to GCS so Render can serve it ──
        report_gcs_key = f"jobs/{job_id}/report.pdf"
        try:
            from app.services.storage import upload_to_storage
            upload_to_storage(report_path, report_gcs_key)
            job_store[job_id]["report_path_gcs"] = report_gcs_key
            logger.info(f"[{job_id}] Uploaded report to GCS: {report_gcs_key}")
        except Exception as up_err:
            logger.warning(f"[{job_id}] GCS report upload failed: {up_err}")

        job_store[job_id].update({
            "status": JobStatus.COMPLETED,
            "progress": 100.0,
            "message": "Fiber Overview Before pipeline completed.",
            "report_path": str(report_path.relative_to(settings.BASE_DIR)),
        })
        logger.info(f"[{job_id}] ✅ Pipeline complete in {total_ms:.0f} ms.")

    except Exception as exc:
        logger.exception(f"[{job_id}] ❌ Pipeline failed: {exc}")
        job_store[job_id].update({
            "status": JobStatus.FAILED,
            "message": "Fiber Overview Before pipeline encountered an error.",
            "error": str(exc),
        })
