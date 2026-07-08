"""
Pipeline Orchestrator — Restored Coordinate & Callout Flow
-----------------------------------------------------------
Key fixes:
  1. Coax pipeline:  encodes callouts with GLOBAL image coords (gx/gy) so
     reporting.py does a clean img→PDF conversion without tile offset math.
  2. Fiber Overview: builds node/splice callout records and passes them to
     generate_final_report so annotations actually appear on the PDF.
"""
import logging, time, cv2, numpy as np, fitz
from pathlib import Path
from app.models.schemas import JobStatus
from app.services.alignment import align_and_pad_maps, iter_tiles, pdf_to_image
from app.services.matching import match_objects
from app.services.rules import RuleEngine
from app.services.reporting import generate_vector_report, generate_final_report
from app.services.fiber_overview import FiberOverviewProcessor
from app.services.storage import download_from_storage, upload_to_storage

logger = logging.getLogger(__name__)


def _pdf_to_bgr(p, dpi=300):
    """Render first page of PDF to BGR numpy array, safely handling RGBA."""
    return pdf_to_image(p, dpi=dpi)


# ─────────────────────────────────────────────────────────────────────────────
# 1. COAX PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_sync(job_id, job_store, settings, detector=None, **kwargs):
    detector = detector or kwargs.get("detector") or kwargs.get("processor")
    if detector is None:
        from app.workers.tasks import _get_detector
        detector = _get_detector(settings)

    def _update(s, p, m):
        if job_id not in job_store:
            job_store[job_id] = {}
        job_store[job_id].update({"status": s, "progress": p, "message": m})
        logger.info(f"[{job_id}] [{p:.0f}%] {m}")

    try:
        job = job_store[job_id]
        out = Path(job["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        dpi = job.get("dpi", 300)

        # Route selected DPI to detector so it loads correct thresholds
        if detector is not None:
            detector.dpi = dpi

        # Download from cloud if GCS endpoints are stored
        if job.get("before_path_gcs"):
            download_from_storage(job["before_path_gcs"], Path(job["before_path"]))
        if job.get("after_path_gcs"):
            download_from_storage(job["after_path_gcs"], Path(job["after_path"]))

        # ── Phase 1: Alignment ────────────────────────────────────────────────
        if job.get("status") == JobStatus.QUEUED:
            _update(JobStatus.ALIGNING, 5.0, "Aligning Coax Maps...")
            fb_raw = _pdf_to_bgr(Path(job["before_path"]), dpi=dpi)
            fa_raw = _pdf_to_bgr(Path(job["after_path"]),  dpi=dpi)
            fb, fa, W = align_and_pad_maps(fb_raw, fa_raw)
            if W is None:
                logger.warning("Alignment failed. Padding maps to max dimensions to prevent crash.")
                h1, w1 = fb.shape[:2]
                h2, w2 = fa.shape[:2]
                h_max, w_max = max(h1, h2), max(w1, w2)
                fb_padded = np.full((h_max, w_max, 3), 255, dtype=np.uint8)
                fa_padded = np.full((h_max, w_max, 3), 255, dtype=np.uint8)
                fb_padded[:h1, :w1] = fb
                fa_padded[:h2, :w2] = fa
                fb, fa = fb_padded, fa_padded
            
            import gc
            del fb_raw, fa_raw
            gc.collect()

            cv2.imwrite(str(out / "aligned_after.png"),  fa)
            cv2.imwrite(str(out / "aligned_before.png"), fb)
            W_inv = np.linalg.inv(W) if W is not None else np.eye(3)
            np.save(str(out / "W_inv.npy"), W_inv)

            # ── Save sample tiles for DPI confirmation preview ────────────────
            # The frontend shows tiles/{before|after}/before_N.png & after_N.png
            tile_size = settings.TILE_SIZE
            sample_indices = []
            
            td_before = out / "tiles" / "before"
            td_after  = out / "tiles" / "after"
            td_before.mkdir(parents=True, exist_ok=True)
            td_after.mkdir(parents=True, exist_ok=True)

            # Find the top 15 densest tiles to ensure they have content (not just blank borders)
            candidate_tiles = []
            for t in iter_tiles(fa, tile_size, settings.TILE_OVERLAP):
                after_tile = t["tile"]
                gray = cv2.cvtColor(after_tile, cv2.COLOR_BGR2GRAY)
                # Count non-white pixels
                density = np.sum(gray < 240) / gray.size
                
                # Only consider tiles with at least some content
                if density > 0.005:
                    candidate_tiles.append((density, t))

            # Sort by density descending and take top 15
            candidate_tiles.sort(key=lambda x: x[0], reverse=True)
            top_candidates = candidate_tiles[:15]

            for density, t in top_candidates:
                s_num = t["index"]
                tx, ty = t["x"], t["y"]
                
                after_tile = t["tile"]
                before_tile = fb[ty:ty+tile_size, tx:tx+tile_size]
                
                before_path = td_before / f"before_{s_num}.png"
                after_path  = td_after  / f"after_{s_num}.png"
                cv2.imwrite(str(before_path), before_tile)
                cv2.imwrite(str(after_path),  after_tile)
                sample_indices.append(s_num)
                logger.info(f"[{job_id}] Saved sample tile pair {s_num} (density: {density:.6f})")

                # Upload tiles to GCS immediately — RunPod containers are stateless.
                # Without this, tiles are lost when the container exits and the frontend shows blanks.
                try:
                    upload_to_storage(before_path, f"jobs/{job_id}/tiles/before/before_{s_num}.png")
                    upload_to_storage(after_path,  f"jobs/{job_id}/tiles/after/after_{s_num}.png")
                    logger.info(f"[{job_id}] Uploaded tile pair {s_num} to GCS.")
                except Exception as _gcs_err:
                    logger.warning(f"[{job_id}] GCS tile upload failed for tile {s_num}: {_gcs_err}")

            if not sample_indices:
                # Fallback: save top-left tile unconditionally
                td_before = out / "tiles" / "before"
                td_after  = out / "tiles" / "after"
                td_before.mkdir(parents=True, exist_ok=True)
                td_after.mkdir(parents=True, exist_ok=True)
                before_path = td_before / "before_1.png"
                after_path  = td_after  / "after_1.png"
                cv2.imwrite(str(before_path), fb[:tile_size, :tile_size])
                cv2.imwrite(str(after_path),  fa[:tile_size, :tile_size])
                logger.info(f"[{job_id}] Saved fallback sample tile pair (1)")
                # Upload fallback tile to GCS as well
                try:
                    upload_to_storage(before_path, f"jobs/{job_id}/tiles/before/before_1.png")
                    upload_to_storage(after_path,  f"jobs/{job_id}/tiles/after/after_1.png")
                except Exception as _gcs_err:
                    logger.warning(f"[{job_id}] GCS fallback tile upload failed: {_gcs_err}")
                sample_indices = [1]

            job_store[job_id].update({
                "status":       JobStatus.AWAITING_DPI_CONFIRM,
                "progress":     15.0,
                "sample_tiles": sample_indices,
            })
            del fb, fa
            gc.collect()
            return

        # ── Phase 2: Detection + Reporting ───────────────────────────────────
        if job.get("status") == JobStatus.PROCESSING:
            _update(JobStatus.PROCESSING, 20.0, "AI analysis running...")
            fa   = cv2.imread(str(out / "aligned_after.png"))
            fb   = cv2.imread(str(out / "aligned_before.png"))
            W_inv = np.load(str(out / "W_inv.npy"))
            re   = RuleEngine()
            callout_records: list[dict] = []
            tile_offsets:    dict       = {}

            # Ensure the detector's thresholds match the job's DPI
            if detector is not None and hasattr(detector, "dpi"):
                detector.dpi = dpi
                logger.info(f"Updated detector DPI to {dpi} for current job.")

            tile_count = 0
            all_tiles = list(iter_tiles(fa, settings.TILE_SIZE, settings.TILE_OVERLAP))
            total_tiles = len(all_tiles)

            for t in all_tiles:
                t_idx = t["index"]
                tx, ty = t["x"], t["y"]
                tile_offsets[t_idx] = (tx, ty)

                tile_count += 1
                if tile_count % 5 == 0 or tile_count == total_tiles:
                    _update(JobStatus.PROCESSING, 20.0 + (tile_count / total_tiles * 50.0), 
                           f"Analysing map... tile {tile_count}/{total_tiles}")

                b_tile = fb[ty: ty + settings.TILE_SIZE, tx: tx + settings.TILE_SIZE]
                a_tile = t["tile"]
                objs_b = detector.detect_objects(b_tile, conf_threshold=0.01)
                objs_a = detector.detect_objects(a_tile, conf_threshold=0.01)
                objs_b = detector.run_ocr_on_objects(b_tile, objs_b)
                objs_a = detector.run_ocr_on_objects(a_tile, objs_a)
                m, r, a = match_objects(objs_b, objs_a, dpi=dpi)
                tile_flagged = False
                for c in re.generate_callouts(m, r, a,
                                                before_node_type=job.get("before_node_type"),
                                                before_node_names=job.get("before_node_names"),
                                                after_node_type=job.get("after_node_type"),
                                                after_node_names=job.get("after_node_names"),
                                                dpi=dpi):
                    # Convert to GLOBAL image coords
                    loc_x, loc_y = c["loc"]
                    gx = tx + loc_x
                    gy = ty + loc_y
                    
                    text_upper = c["text"].upper()
                    is_flagged = "WARNING" in text_upper or text_upper in ["G", "POWERBLOCK", "ADD POWER BLOCK", "REMOVE POWER BLOCK"]
                    if is_flagged:
                        tile_flagged = True
                    
                    callout_records.append({
                        "tile_idx": t_idx,
                        "gx": gx,
                        "gy": gy,
                        "lx": loc_x,  # kept for compat
                        "ly": loc_y,
                        "text": c["text"],
                        "type": "FLAGGED" if is_flagged else "NORMAL"
                    })

                # Write flagged tiles to disk on the fly so the frontend can load them
                if tile_flagged:
                    td_before = out / "tiles" / "before"
                    td_after  = out / "tiles" / "after"
                    td_before.mkdir(parents=True, exist_ok=True)
                    td_after.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(td_before / f"before_{t_idx}.png"), b_tile)
                    cv2.imwrite(str(td_after  / f"after_{t_idx}.png"),  a_tile)

                # Periodic memory cleanup to prevent OOM
                if tile_count % 10 == 0:
                    import gc
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass

            # Save state and halt for human review
            # Extract unique tile indices that have flagged callouts
            flagged_indices = list({c["tile_idx"] for c in callout_records if c.get("type") == "FLAGGED"})

            # Save state and halt for human review
            job_store[job_id].update({
                "status": JobStatus.AWAITING_REVIEW,
                "progress": 80.0,
                "flagged_tiles": flagged_indices,
                "all_callouts": callout_records
            })
            del fa, fb
            import gc
            gc.collect()
            return

        if job.get("status") == JobStatus.REPORTING:
            _update(JobStatus.REPORTING, 85.0, "Generating vector report...")
            fa   = cv2.imread(str(out / "aligned_after.png"))
            W_inv = np.load(str(out / "W_inv.npy"))
            
            # Apply user overrides
            callout_records = job.get("all_callouts", [])
            overrides = job.get("all_callouts_visible", [])
            
            # overrides is a list of {tileIdx, action, newText}
            for override in overrides:
                target_idx = override.get("tileIdx")
                action = override.get("action")
                new_text = override.get("newText")
                
                for c in callout_records:
                    if c["tile_idx"] == target_idx:
                        if action == "REMOVE":
                            c["removed"] = True
                        elif action == "RENAME" and new_text:
                            c["text"] = new_text

            final_callouts = [c for c in callout_records if not c.get("removed")]

            # Inject the Design Note for Coax After result map
            final_callouts.append({
                "tile_idx": 0,
                "gx": fa.shape[1] / 2.0,  # Start search from center of the map
                "gy": fa.shape[0] / 2.0,
                "lx": 0,
                "ly": 0,
                "text": "DESIGN NOTE: CHECK ALL ACTIVES AND ENSURE THEY\nHAVE BEEN REBALANCED PROPERLY",
                "is_design_note": True,
                "type": "NORMAL"
            })

            tile_offsets = {}
            for t in iter_tiles(fa, settings.TILE_SIZE, settings.TILE_OVERLAP):
                tile_offsets[t["index"]] = (t["x"], t["y"])

            generate_vector_report(
                after_pdf_path=Path(job["after_path"]),
                callout_records=final_callouts,
                tile_offsets=tile_offsets,
                W_inv=W_inv,
                output_path=out / "report.pdf",
                dpi=dpi,
                survey_image_path=job.get("survey_image_path"),
                title_box_data={
                    "prism_id":   job.get("title_box", {}).get("prism_id", ""),
                    "node_name":  job.get("title_box", {}).get("node_name", ""),
                    "instance":   job.get("title_box", {}).get("instance", ""),
                    "map_type":   job.get("title_box", {}).get("map_type", "AFTER"),
                    "page_count": 1,
                },
                title_font_size=24,
            )
            logger.info(f"[{job_id}] Final report saved to: {out / 'report.pdf'}")
            
            # Upload report to GCS
            report_path = out / "report.pdf"
            gcs_uri = upload_to_storage(report_path, f"jobs/{job_id}/report.pdf")
            if gcs_uri:
                job_store[job_id]["report_path_gcs"] = f"jobs/{job_id}/report.pdf"

            job_store[job_id].update({
                "status":      JobStatus.COMPLETED,
                "progress":    100.0,
                "callouts":    final_callouts,
                "report_path": str((out / "report.pdf").relative_to(settings.BASE_DIR)),
            })

    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        _update(JobStatus.FAILED, 0, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 2. FIBER OVERVIEW PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_fiber_overview_pipeline(job_id, job_store, settings, processor=None, **kwargs):
    """
    Runs node + cable detection on the Fiber Overview map,
    builds callout records from node / splice-can results,
    and renders them via generate_final_report.
    """
    processor = processor or kwargs.get("processor")
    if processor is None:
        from app.workers.tasks import _get_overview_processor
        processor = _get_overview_processor(settings)

    def _update(s, p, m):
        job_store[job_id].update({"status": s, "progress": p, "message": m})
        logger.info(f"[{job_id}] [{p:.0f}%] {m}")

    try:
        job = job_store[job_id]
        out = Path(job["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        dpi = job.get("dpi", 300)
        title_box = job.get("title_box", {})

        _update(JobStatus.PROCESSING, 20.0, "Rendering fiber overview map...")
        logger.info(f"[{job_id}] Rendering fiber overview map...")

        # Download from cloud if GCS endpoints are stored
        if job.get("pdf_path_gcs"):
            download_from_storage(job["pdf_path_gcs"], Path(job["pdf_path"]))

        # ── Render PDF to image ───────────────────────────────────────────────
        pdf_path = Path(job["pdf_path"])
        img_bgr = pdf_to_image(pdf_path, dpi=dpi)

        # ── Node detection ────────────────────────────────────────────────────
        _update(JobStatus.PROCESSING, 40.0, "Detecting fiber node...")
        logger.info(f"[{job_id}] Detecting fiber node...")
        bbox, center, conf = processor.detect_node(img_bgr)

        callout_records: list[dict] = []
        scale = 72.0 / dpi  # image px → PDF pts

        if bbox and center:
            nx, ny = center
            # NODE callout
            callout_records.append({
                "gx": float(nx), "gy": float(ny),
                "text": "NODE",
            })

            # ── Port/splice-can tracing ───────────────────────────────────────
            _update(JobStatus.PROCESSING, 60.0, "Tracing cable skeleton...")
            logger.info(f"[{job_id}] Tracing cable skeleton...")
            skeleton = processor.extract_cable_skeleton(img_bgr, bbox)
            if skeleton is not None:
                port_pos = processor.find_port_position(skeleton, bbox)
                if port_pos:
                    px, py = port_pos
                    
                    is_connected    = job.get("is_connected", True)
                    hub_name        = job.get("hub_name", "")
                    port_name       = job.get("port_name", "")
                    splice_can_name = job.get("splice_can_name", "")
                    
                    if is_connected:
                        port_text = f"HUB : {hub_name}\nPORT/PANEL : {port_name}"
                    else:
                        port_text = (
                            f"TRACE STOPS AT RAW CAN ({splice_can_name}) ; "
                            "EXISTING SPLICING UNAVAILABLE , A CAN AUDIT REQUIRED FOR VERIFICATION"
                        )
                    
                    callout_records.append({
                        "gx": float(px), "gy": float(py),
                        "text": port_text,
                    })

        _update(JobStatus.REPORTING, 80.0, "Rendering PDF report...")

        report_path = out / "report.pdf"
        generate_final_report(
            pdf_path=pdf_path,
            callouts=callout_records,
            output_path=report_path,
            dpi=dpi,
            survey_image_path=job.get("survey_image_path"),
            title_box_data={
                "prism_id":   title_box.get("prism_id", ""),
                "node_name":  title_box.get("node_name", ""),
                "instance":   title_box.get("instance", ""),
                "map_type":   "FIBER OVERVIEW",
                "page_count": 1,
            },
            title_font_size=14,
        )
        logger.info(f"[{job_id}] Fiber Overview report saved to: {report_path}")

        # Upload report to GCS
        gcs_uri = upload_to_storage(report_path, f"jobs/{job_id}/report.pdf")
        if gcs_uri:
            job_store[job_id]["report_path_gcs"] = f"jobs/{job_id}/report.pdf"

        job_store[job_id].update({
            "status":      JobStatus.COMPLETED,
            "progress":    100.0,
            "message":     "Fiber Overview report complete.",
            "callouts":    callout_records,
            "report_path": str(report_path.relative_to(settings.BASE_DIR)),
        })

    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        if job_id in job_store:
            job_store[job_id].update({"status": JobStatus.FAILED, "message": str(e)})
