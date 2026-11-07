"""
Stateless Service for generating the 'Coax Before' map overview.
Ports the overlay_tool.py logic into the API worker architecture.
Finds the whitest map corner automatically, stamps Survey Image + Title Box.
No callout or ML logic involved.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import fitz
import numpy as np

from app.core.config import Settings
from app.models.schemas import JobStatus
from app.services.storage import download_from_storage, upload_to_storage

logger = logging.getLogger(__name__)


def _get_best_corner(
    page: fitz.Page,
    ss_width_pts: float,
    full_overlay_h: float,
    margin: float = 30.0,
) -> tuple[float, float, bool]:
    """
    Analyzes the first page raster to find the corner with the highest
    average brightness (whitest area). Returns (x, y, needs_extension).
    """
    pix = page.get_pixmap(dpi=150)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

    if pix.n == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    h_img, w_img = gray.shape
    pw, ph = page.rect.width, page.rect.height

    corners = {
        "TL": (margin, margin),
        "TR": (pw - ss_width_pts - margin, margin),
        "BL": (margin, ph - full_overlay_h - margin),
        "BR": (pw - ss_width_pts - margin, ph - full_overlay_h - margin),
    }

    THRESHOLD = 252  # very strict white check
    brightness_vals: dict[str, float] = {}
    clear_corners: list[str] = []
    scale = 150.0 / 72.0

    for name, (cx, cy) in corners.items():
        x1 = max(0, int(cx * scale))
        y1 = max(0, int(cy * scale))
        x2 = min(w_img, int((cx + ss_width_pts) * scale))
        y2 = min(h_img, int((cy + full_overlay_h) * scale))
        roi = gray[y1:y2, x1:x2]
        if roi.size > 0:
            avg_b = float(np.mean(roi))
            min_b = float(np.min(roi))
            brightness_vals[name] = avg_b
            if min_b >= THRESHOLD:
                clear_corners.append(name)

    logger.debug(f"Corner brightness: {brightness_vals}")
    logger.debug(f"Clear corners (min >= {THRESHOLD}): {clear_corners}")

    if not clear_corners:
        logger.info("No clear corners found — triggering map extension mode.")
        return (margin, -full_overlay_h - margin, True)

    best = max(clear_corners, key=lambda n: brightness_vals[n])
    cx, cy = corners[best]
    return (cx, cy, False)


def _overlay_on_pdf(
    pdf_path: Path,
    screenshot_path: Path | None,
    prism_id: str,
    node_name: str,
    instance: str,
    map_type: str,
    output_path: Path,
) -> None:
    """
    Core overlay function. Uses unified 4-corner whitest placement with
    movable Stamp survey image and movable FreeText title box.
    """
    from app.services.reporting import _draw_legend_stack

    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count

    title_box_data = {
        "prism_id": prism_id,
        "node_name": node_name,
        "instance": instance,
        "map_type": map_type if map_type else "BEFORE PRINT",
        "page_count": total_pages,
    }

    for pg_idx in range(total_pages):
        page = doc[pg_idx]
        _draw_legend_stack(
            page=page,
            survey_image_path=screenshot_path if pg_idx == 0 else None,
            title_box_data=title_box_data,
            title_font_size=24,
            include_legend=False,
            page_num=pg_idx + 1,
            total_pages=total_pages,
        )

    # Patch appearance streams for title box font preservation
    for page_idx in range(doc.page_count):
        p = doc[page_idx]
        for a in list(p.annots()):
            if a.type[0] == 2:
                xref = a.xref
                doc.xref_set_key(xref, "F", "4")
                doc.xref_set_key(xref, "BS", "<< /W 2.5 /S /S >>")
                doc.xref_set_key(xref, "C", "[1 0 0]")
                doc.xref_set_key(xref, "IC", "[1 1 0]")
                doc.xref_set_key(xref, "DA", "(0 0 0 rg /Helv 24 Tf)")
                doc.xref_set_key(
                    xref, "DS",
                    "(font: Helv 24pt; color: #000000; background-color: #FFFF00; border: 2.5pt solid #FF0000;)"
                )

    doc.save(str(output_path), deflate=True, clean=False)
    doc.close()


def run_coax_before_pipeline(
    job_id: str,
    job_store: dict,
    settings: Settings,
) -> None:
    """
    Stateless processing for the Coax Before map.
    Stamps Survey Image + Title Box using smart whitest-corner placement.
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
        survey_image_path = job.get("survey_image_path")
        title_box = job.get("title_box", {})
        prism_id = title_box.get("prism_id", "")
        node_name = title_box.get("node_name", "")
        instance = title_box.get("instance", "")
        map_type = title_box.get("map_type", "BEFORE PRINT")

        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        job_store[job_id]["stage_times"] = {}
        t0 = time.perf_counter()

        # ── Download from GCS if running on RunPod (file won't exist locally) ──
        if not pdf_path.exists():
            gcs_key = job.get("pdf_path_gcs") or job.get("before_path_gcs")
            if gcs_key:
                try:
                    _update(JobStatus.PROCESSING, 5, "Downloading PDF from cloud storage...")
                    download_from_storage(gcs_key, pdf_path)
                    logger.info(f"[{job_id}] Downloaded PDF from GCS: {gcs_key}")
                except Exception as dl_err:
                    raise FileNotFoundError(
                        f"PDF not found locally and GCS download failed: {dl_err}"
                    )
            else:
                raise FileNotFoundError(
                    f"PDF not found at {pdf_path} and no GCS path available."
                )

        # ── Download survey image from GCS if needed ──
        if survey_image_path:
            survey_path_obj = Path(survey_image_path)
            if not survey_path_obj.exists():
                survey_gcs = job.get("survey_image_path_gcs")
                if survey_gcs:
                    try:
                        survey_path_obj.parent.mkdir(parents=True, exist_ok=True)
                        download_from_storage(survey_gcs, survey_path_obj)
                        logger.info(f"[{job_id}] Downloaded survey image from GCS.")
                    except Exception:
                        survey_image_path = None  # proceed without it

        _update(JobStatus.PROCESSING, 20, "Analysing map corners...")
        report_path = output_dir / "report.pdf"

        _overlay_on_pdf(
            pdf_path=pdf_path,
            screenshot_path=Path(survey_image_path) if survey_image_path else None,
            prism_id=prism_id,
            node_name=node_name,
            instance=instance,
            map_type=map_type,
            output_path=report_path,
        )
        t0 = _record("REPORTING", t0)

        _update(JobStatus.PROCESSING, 90, "Saving annotated PDF...")
        total_ms = (time.perf_counter() - job_start) * 1000
        job_store[job_id]["stage_times"]["total_ms"] = round(total_ms, 1)

        # ── Upload report to GCS so Render can serve it ──
        report_gcs_key = f"jobs/{job_id}/report.pdf"
        try:
            upload_to_storage(report_path, report_gcs_key)
            job_store[job_id]["report_path_gcs"] = report_gcs_key
            logger.info(f"[{job_id}] Uploaded report to GCS: {report_gcs_key}")
        except Exception as up_err:
            logger.warning(f"[{job_id}] GCS report upload failed: {up_err}")

        job_store[job_id].update({
            "status": JobStatus.COMPLETED,
            "progress": 100.0,
            "message": "Coax Before pipeline completed.",
            "report_path": str(report_path.relative_to(settings.BASE_DIR)),
        })
        logger.info(f"[{job_id}] ✅ Coax Before pipeline complete in {total_ms:.0f} ms.")

    except Exception as exc:
        logger.exception(f"[{job_id}] ❌ Coax Before pipeline failed: {exc}")
        job_store[job_id].update({
            "status": JobStatus.FAILED,
            "message": "Coax Before pipeline encountered an error.",
            "error": str(exc),
        })

