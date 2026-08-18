"""
Standalone Inference & Evaluation for Repaired Model Weights
------------------------------------------------------------
Uses the isolated copy of model weights: model_training/repaired_weights/best_repaired.pt
Original backend/model_weights/ files remain completely untouched.

Features:
- High-resolution sliding window tile processing (1280x1280) with NMS merging.
- Node containment suppression (removes inner symbols inside Node boxes).
- Excludes solid black map pole triangles/rectangles from Terminators.
- Correctly identifies Splice blocks ([B] and [S]) as Class 14 (Splice).
"""
import sys
import os
import re
from pathlib import Path
import cv2
import fitz
import numpy as np

def run_repaired_inference(pdf_path_str: str):
    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        print(f"❌ Input PDF not found: {pdf_path}")
        return

    weights_path = Path("/app/model_training/repaired_weights/best_repaired.pt")
    if not weights_path.exists():
        weights_path = Path(__file__).parent / "repaired_weights" / "best_repaired.pt"

    print(f"🤖 Loading Isolated Repaired Model Weights: {weights_path}")
    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    doc = fitz.open(pdf_path)
    print(f"📄 Processing Page 1 of PDF Map: {pdf_path.name}")
    page = doc[0]

    dpi = 300
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1]
    h, w = img_bgr.shape[:2]
    print(f"📐 Full Map Rendered Resolution: {w}x{h} px")

    # PyMuPDF text search for Splice Blocks [B] and [S]
    splice_instances = []
    for word_info in page.get_text("words"):
        w_text = word_info[4].strip()
        if w_text in ["B", "S", "[B]", "[S]", "SPLICE"]:
            r_x0, r_y0, r_x1, r_y1 = word_info[:4]
            cx_px = (r_x0 + r_x1) / 2.0 * zoom
            cy_px = (r_y0 + r_y1) / 2.0 * zoom
            splice_instances.append((cx_px, cy_px))

    # PyMuPDF search for TAP (EQZ) / CEX Taps
    eqz_text_instances = []
    for search_str in ["EQZ", "CE-", "CE ", "CE_"]:
        for r in page.search_for(search_str):
            cx_px = (r.x0 + r.x1) / 2.0 * zoom
            cy_px = (r.y0 + r.y1) / 2.0 * zoom
            eqz_text_instances.append((cx_px, cy_px))

    tile_size = 1280
    overlap = 256
    step = tile_size - overlap

    all_detections = []
    NODE_CLASSES = {0, 2, 5, 6}  # 1x4 Node, 2x2 Node, 3x3 Node, 4x4 Node

    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = img_bgr[y:min(y+tile_size, h), x:min(x+tile_size, w)]
            th, tw = tile.shape[:2]

            if th < tile_size or tw < tile_size:
                padded = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
                padded[:th, :tw] = tile
                tile = padded

            results = model(tile, conf=0.15, verbose=False)[0]

            if hasattr(results, "boxes") and results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])

                    x1_t, y1_t, x2_t, y2_t = box.xyxy[0].cpu().numpy()
                    gx1, gy1 = x1_t + x, y1_t + y
                    gx2, gy2 = x2_t + x, y2_t + y
                    bw_px, bh_px = gx2 - gx1, gy2 - gy1

                    # Filter out tiny pole/house numbers misclassified as taps
                    if cls_id == 19 and (bw_px < 42 or bh_px < 42) and conf_val < 0.35:
                        continue

                    # Filter out solid black triangles and rectangles from Terminators (Class 20)
                    if cls_id == 20:
                        cx_int, cy_int = int((x1_t + x2_t)/2.0), int((y1_t + y2_t)/2.0)
                        crop = tile[max(0, cy_int-15):min(tile_size, cy_int+15), max(0, cx_int-15):min(tile_size, cx_int+15)]
                        if crop.size > 0 and np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) < 70.0:
                            continue

                    all_detections.append({
                        "cls_id": cls_id,
                        "class_name": model.names[cls_id],
                        "conf": conf_val,
                        "box": [gx1, gy1, gx2, gy2]
                    })

    # Perform Non-Maximum Suppression (NMS) across overlapping tile detections
    def ioa_nms(dets, iou_thresh=0.45):
        if not dets: return []
        boxes = np.array([d["box"] for d in dets])
        scores = np.array([d["conf"] for d in dets])
        cls_ids = np.array([d["cls_id"] for d in dets])

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w_int = np.maximum(0.0, xx2 - xx1)
            h_int = np.maximum(0.0, yy2 - yy1)
            inter = w_int * h_int

            same_cls = (cls_ids[order[1:]] == cls_ids[i])
            ovr = np.zeros_like(inter)
            ovr[same_cls] = inter[same_cls] / (areas[i] + areas[order[1:][same_cls]] - inter[same_cls])

            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]

        return [dets[k] for k in keep]

    merged_dets = ioa_nms(all_detections)

    # Node containment suppression (Remove inner symbols inside Node boxes)
    node_boxes = [d["box"] for d in merged_dets if d["cls_id"] in NODE_CLASSES]
    final_dets = []
    for d in merged_dets:
        if d["cls_id"] in NODE_CLASSES:
            final_dets.append(d)
        else:
            cx, cy = (d["box"][0] + d["box"][2])/2.0, (d["box"][1] + d["box"][3])/2.0
            inside_node = any(nb[0] <= cx <= nb[2] and nb[1] <= cy <= nb[3] for nb in node_boxes)
            if not inside_node:
                final_dets.append(d)

    # Check for TAP (EQZ) CEX Taps
    cex_tap_count = 0
    for d in final_dets:
        if d["cls_id"] == 19:
            cx, cy = (d["box"][0] + d["box"][2])/2.0, (d["box"][1] + d["box"][3])/2.0
            if any(abs(cx - eq_x) < 80 and abs(cy - eq_y) < 80 for eq_x, eq_y in eqz_text_instances):
                d["class_name"] = "TAP (EQZ) - ADD CE-XX"
                cex_tap_count += 1

    # Inject Splice blocks [B] / [S]
    splice_count = 0
    for sp_x, sp_y in splice_instances:
        # Check if already detected or inside node
        inside_node = any(nb[0] <= sp_x <= nb[2] and nb[1] <= sp_y <= nb[3] for nb in node_boxes)
        if not inside_node:
            # Override any Tap at this location with Splice
            final_dets = [d for d in final_dets if not (d["cls_id"] == 19 and abs((d["box"][0]+d["box"][2])/2.0 - sp_x) < 40 and abs((d["box"][1]+d["box"][3])/2.0 - sp_y) < 40)]
            final_dets.append({
                "cls_id": 14,
                "class_name": "Splice",
                "conf": 0.99,
                "box": [sp_x - 30, sp_y - 30, sp_x + 30, sp_y + 30]
            })
            splice_count += 1

    # Print Summary Table
    counts = {}
    for d in final_dets:
        cname = d["class_name"]
        counts[cname] = counts.get(cname, 0) + 1

    print("\n" + "="*50)
    print("🏆 REPAIRED MODEL SLIDING-WINDOW DETECTION RESULTS")
    print(f"📄 Input File: {pdf_path.name}")
    print(f"🎯 Total Detected Symbols: {len(final_dets)}")
    print("="*50)
    print(f"{'Symbol Class':<25} | {'Count':<10}")
    print("-" * 40)
    for cname in sorted(counts.keys()):
        print(f"{cname:<25} | {counts[cname]:<10}")
    print("="*50)

    # Draw Bounding Boxes on Full Map Image
    annotated_img = img_bgr.copy()
    for d in final_dets:
        b = [int(v) for v in d["box"]]
        cname = d["class_name"]
        cv2.rectangle(annotated_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
        cv2.putText(annotated_img, cname, (b[0], max(15, b[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"repaired_annotated_{pdf_path.stem}.jpg"
    cv2.imwrite(str(out_file), annotated_img)
    print(f"\n🖼️ Full-Resolution Annotated Map Saved To:\n   [out_file] -> {out_file}\n")

if __name__ == "__main__":
    pdf_input = sys.argv[1] if len(sys.argv) > 1 else "/app/model_training/raw_samples/4358271_AFTER_COAX.pdf"
    run_repaired_inference(pdf_input)
