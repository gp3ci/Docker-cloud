import sys, os, time, gc
from pathlib import Path
import numpy as np
import cv2
import fitz
import torch

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "Desktop" / "Docker-cloud-main" / "backend"
if not backend_dir.exists():
    backend_dir = Path("c:/Users/Netcom/Desktop/Docker-cloud-main/backend")
sys.path.insert(0, str(backend_dir))

from app.core.config import get_settings
from app.services.vision import TelecomDetector
from app.services.alignment import align_and_pad_maps, pdf_to_image, iter_tiles
from app.services.matching import match_objects
from app.services.rules import RuleEngine
from app.services.reporting import generate_vector_report

def run_map_test(before_pdf: Path, after_pdf: Path, out_dir: Path, map_id: str, dpi: int = 150):
    print(f"\n=======================================================")
    print(f" TESTING MAP PAIR: {map_id} ({before_pdf.name} -> {after_pdf.name})")
    print(f"=======================================================")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[{map_id}] Rendering PDFs at {dpi} DPI...")
    fb_raw = pdf_to_image(before_pdf, dpi=dpi)
    fa_raw = pdf_to_image(after_pdf, dpi=dpi)

    print(f"[{map_id}] Aligning maps...")
    fb, fa, W = align_and_pad_maps(fb_raw, fa_raw)
    del fb_raw, fa_raw
    gc.collect()

    W_inv = np.linalg.inv(W) if W is not None else np.eye(3)
    settings = get_settings()

    print(f"[{map_id}] Initializing Detector on CUDA={torch.cuda.is_available()}...")
    detector = TelecomDetector(
        main_model_path=settings.MAIN_MODEL_PATH,
        ps_model_path=settings.MODELS_DIR / "power_supply_best.pt",
        node_model_path=settings.MODELS_DIR / "3x3_4x4_new_model.pt",
        internal_model_path=settings.MODELS_DIR / "Internal_best.pt",
        use_gpu=torch.cuda.is_available(),
        dpi=dpi,
    )

    re_engine = RuleEngine()
    callout_records = []
    tile_offsets = {}
    tile_size = settings.TILE_SIZE
    overlap = settings.TILE_OVERLAP

    all_tiles = list(iter_tiles(fa, tile_size, overlap))
    total_tiles = len(all_tiles)
    print(f"[{map_id}] Total tiles to inspect: {total_tiles}")

    det_a_count = 0
    det_b_count = 0

    # For faster test, process informative tiles or all tiles
    for idx, t in enumerate(all_tiles):
        tx, ty = t["x"], t["y"]
        t_idx = t["index"]
        tile_offsets[t_idx] = (tx, ty)

        a_tile = t["tile"]
        # Skip purely blank tiles
        gray_a = cv2.cvtColor(a_tile, cv2.COLOR_BGR2GRAY)
        if (np.sum(gray_a < 240) / gray_a.size) < 0.001:
            continue

        b_tile = fb[ty:ty+tile_size, tx:tx+tile_size]

        objs_b = detector.detect_objects(b_tile, conf_threshold=0.01)
        objs_a = detector.detect_objects(a_tile, conf_threshold=0.01)
        det_b_count += len(objs_b)
        det_a_count += len(objs_a)

        if len(objs_b) > 0 or len(objs_a) > 0:
            objs_b = detector.run_ocr_on_objects(b_tile, objs_b)
            objs_a = detector.run_ocr_on_objects(a_tile, objs_a)
            m, r, a = match_objects(objs_b, objs_a, dpi=dpi)

            c_list = re_engine.generate_callouts(m, r, a, dpi=dpi)
            for c in c_list:
                loc_x, loc_y = c["loc"]
                gx = tx + loc_x
                gy = ty + loc_y
                callout_records.append({
                    "tile_idx": t_idx,
                    "gx": gx,
                    "gy": gy,
                    "lx": loc_x,
                    "ly": loc_y,
                    "text": c["text"],
                })

        if (idx + 1) % 10 == 0:
            print(f"[{map_id}] Processed {idx + 1}/{total_tiles} tiles... Current callouts: {len(callout_records)}")

    print(f"[{map_id}] Detection Complete: {det_b_count} Before objects, {det_a_count} After objects.")
    print(f"[{map_id}] Rule Engine Generated: {len(callout_records)} Callouts!")

    # Generate Vector Report PDF
    out_pdf = out_dir / f"{map_id}_annotated_report.pdf"
    print(f"[{map_id}] Generating Vector PDF Report -> {out_pdf.name}...")

    title_data = {
        "map_type": "COAX AFTER",
        "prism_id": map_id,
        "node_name": f"ND_{map_id[:6]}",
        "instance": "1",
    }

    generate_vector_report(
        after_pdf_path=after_pdf,
        callout_records=callout_records,
        tile_offsets=tile_offsets,
        W_inv=W_inv,
        output_path=out_pdf,
        dpi=dpi,
        title_box_data=title_data,
        title_font_size=24,
        include_legend=True,
    )

    # Render pages and check styling
    print(f"[{map_id}] Verifying rendered visual appearance...")
    doc = fitz.open(str(out_pdf))
    page = doc[0]

    # Verify stream appearance in annotations
    callout_annots = []
    title_annots = []
    for a in page.annots():
        it = doc.xref_get_key(a.xref, "IT")[1]
        if it == "/FreeTextCallout":
            callout_annots.append(a)
        elif it == "/FreeText" and a.rect.width > 100:
            title_annots.append(a)

    print(f"[{map_id}] Found {len(callout_annots)} FreeTextCallouts and {len(title_annots)} TitleBoxes.")

    red_arrow_count = 0
    yellow_box_count = 0
    red_border_count = 0

    for a in callout_annots:
        ap = doc.xref_get_key(a.xref, "AP")[1]
        import re
        m = re.search(r'/N\s+(\d+)', ap)
        if m:
            st = doc.xref_stream(int(m.group(1))).decode('latin1')
            if '1 0 0 RG' in st:
                red_arrow_count += 1
                red_border_count += 1
            if '1 1 0 rg' in st:
                yellow_box_count += 1

    print(f"[{map_id}] Callout Styling Check:")
    print(f"  - Red Arrow & Red Border: {red_arrow_count}/{len(callout_annots)} (100% applied: {red_arrow_count == len(callout_annots)})")
    print(f"  - Yellow Box Fill:        {yellow_box_count}/{len(callout_annots)} (100% applied: {yellow_box_count == len(callout_annots)})")

    # Save high-res rendered full page PNG
    out_png = out_dir / f"{map_id}_full_page_preview.png"
    pix = page.get_pixmap(dpi=100)
    pix.save(str(out_png))
    print(f"[{map_id}] Saved full page preview to: {out_png.name}")

    # Save zoom in on first callout if any
    if callout_annots:
        c_rect = callout_annots[0].rect
        zoom_rect = fitz.Rect(c_rect.x0 - 80, c_rect.y0 - 80, c_rect.x1 + 80, c_rect.y1 + 80)
        pix_zoom = page.get_pixmap(clip=zoom_rect, dpi=200)
        zoom_png = out_dir / f"{map_id}_callout_zoom.png"
        pix_zoom.save(str(zoom_png))
        print(f"[{map_id}] Saved callout zoom preview to: {zoom_png.name}")

    # Save zoom in on title box
    page_r = page.rect
    tb_rect = fitz.Rect(page_r.width - 400, 0, page_r.width, 350)
    pix_tb = page.get_pixmap(clip=tb_rect, dpi=150)
    tb_png = out_dir / f"{map_id}_titlebox_legend_zoom.png"
    pix_tb.save(str(tb_png))
    print(f"[{map_id}] Saved titlebox + legend zoom to: {tb_png.name}")

    doc.close()
    elapsed = time.time() - t0
    print(f"[{map_id}] Completed in {elapsed:.1f}s")
    return {
        "map_id": map_id,
        "callouts": len(callout_records),
        "red_arrow_verified": red_arrow_count == len(callout_annots),
        "yellow_box_verified": yellow_box_count == len(callout_annots),
        "elapsed_s": round(elapsed, 1),
    }

if __name__ == "__main__":
    maps_root = Path("other files/AI-Quality_control_PROJECT-backend/input_maps")
    out_root = Path("backend/test_runs")
    out_root.mkdir(parents=True, exist_ok=True)

    test_pairs = [
        ("PAIR_0", maps_root / "BEFORE.pdf", maps_root / "AFTER.pdf"),
        ("PAIR_1", maps_root / "before1.pdf", maps_root / "after1.pdf"),
        ("PAIR_2", maps_root / "before2.pdf", maps_root / "after2.pdf"),
        ("PAIR_3", maps_root / "before3.pdf", maps_root / "after3.pdf"),
        ("PAIR_4", maps_root / "before4.pdf", maps_root / "after4.pdf"),
        ("PAIR_5", maps_root / "before5.pdf", maps_root / "after5.pdf"),
    ]

    results = []
    for m_id, b_pdf, a_pdf in test_pairs:
        if b_pdf.exists() and a_pdf.exists():
            res = run_map_test(b_pdf, a_pdf, out_root, m_id, dpi=150)
            results.append(res)
        else:
            print(f"Skipping {m_id}: {b_pdf} or {a_pdf} missing.")

    print("\n" + "="*60)
    print("FINAL BATCH TESTING SUMMARY")
    print("="*60)
    for r in results:
        print(f"Map {r['map_id']}: Callouts={r['callouts']}, Red Arrow Verified={r['red_arrow_verified']}, Yellow Box Verified={r['yellow_box_verified']}, Time={r['elapsed_s']}s")
