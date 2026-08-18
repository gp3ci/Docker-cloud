"""
Standalone Telecom YOLO Model Tester (Sliding Window Tile Detector)
-------------------------------------------------------------------
Runs full-resolution 1280x1280 sliding window tile detection with Non-Maximum
Suppression (NMS) merging across the entire map sheet.

Usage:
    python model_training/test_standalone_model.py [path_to_map.pdf_or_image]
"""
import sys
import os
from pathlib import Path
import fitz
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def apply_nms(detections, iou_thresh=0.35):
    if not detections:
        return []
    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    kept = []
    for d in detections:
        overlap = False
        for k in kept:
            if d["class_name"] == k["class_name"] and box_iou(d["bbox"], k["bbox"]) > iou_thresh:
                overlap = True
                break
        if not overlap:
            kept.append(d)
    return kept

def test_model(input_file=None):
    base_dir = Path(__file__).parent
    output_dir = base_dir / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = base_dir / "runs" / "detect" / "symbol_detection_yolov8n_retrained" / "weights" / "best.pt"
    if not weights_path.exists():
        weights_path = base_dir.parent / "backend" / "model_weights" / "best.pt"
    if not weights_path.exists():
        weights_path = Path("/app/model_weights/best.pt")

    if not weights_path.exists():
        print(f"❌ Model weights not found at {weights_path}")
        return

    if not input_file:
        raw_samples = list((base_dir / "raw_samples").glob("*.pdf"))
        if raw_samples:
            input_file = raw_samples[0]
        else:
            print("❌ Please provide a PDF or image file path to test.")
            return

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return

    print(f"🤖 Loading Retrained Model Weights: {weights_path}")
    model = YOLO(str(weights_path))

    # Convert Page 1 of PDF map to 300 DPI image
    eqz_instances = []
    if input_path.suffix.lower() == ".pdf":
        print(f"📄 Processing Page 1 of PDF Map: {input_path.name}")
        doc = fitz.open(input_path)
        page = doc[0]
        dpi = 300
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1]

        # PyMuPDF search for EQZ / CE text
        for search_str in ["EQZ", "CE-", "CE ", "CE_"]:
            text_rects = page.search_for(search_str)
            for r in text_rects:
                cx_px = (r.x0 + r.x1) / 2.0 * zoom
                cy_px = (r.y0 + r.y1) / 2.0 * zoom
                eqz_instances.append((cx_px, cy_px))
    else:
        print(f"🖼️ Processing Image File: {input_path.name}")
        img_bgr = cv2.imread(str(input_path))

    h, w = img_bgr.shape[:2]
    print(f"📐 Full Resolution: {w}x{h} px")

    # Sliding Window Tiling (1280x1280 with 256px overlap)
    tile_size = 1280
    overlap = 256
    step = tile_size - overlap

    raw_detections = []
    print("🔍 Sliding Window Tile Processing across full map sheet...")

    ALLOWED_CLASSES = {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21}
    NODE_CLASSES = {"1x4 Node", "2x2 Node", "3x3 Node", "4x4 Node"}

    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = img_bgr[y:min(y+tile_size, h), x:min(x+tile_size, w)]
            th, tw = tile.shape[:2]

            if th < tile_size or tw < tile_size:
                padded = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
                padded[:th, :tw] = tile
                tile = padded

            # Run YOLO prediction on tile at conf=0.35 to filter out background noise
            results = model(tile, conf=0.35, verbose=False)[0]

            if hasattr(results, "boxes") and results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    cls_name = model.names[cls_id]

                    if cls_id not in ALLOWED_CLASSES or conf_val < 0.35:
                        continue

                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    bw_px = bx2 - bx1
                    bh_px = by2 - by1

                    # House Number Filter: Small round circles (<42px) are house numbers, NOT taps!
                    if cls_id == 19 and (bw_px < 42 or bh_px < 42) and conf_val < 0.35:
                        continue

                    gx1 = int(x + bx1)
                    gy1 = int(y + by1)
                    gx2 = int(x + bx2)
                    gy2 = int(y + by2)

                    raw_detections.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf_val,
                        "bbox": (gx1, gy1, gx2, gy2),
                        "center": ((gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0)
                    })

    # Apply NMS merging
    merged_detections = apply_nms(raw_detections, iou_thresh=0.35)

    # Node Containment Suppression (Remove symbols inside Node boxes)
    node_boxes = [d for d in merged_detections if d["class_name"] in NODE_CLASSES]
    final_detections = []

    for d in merged_detections:
        if d["class_name"] in NODE_CLASSES:
            final_detections.append(d)
        else:
            cx, cy = d["center"]
            inside_node = any(nb["bbox"][0] <= cx <= nb["bbox"][2] and nb["bbox"][1] <= cy <= nb["bbox"][3] for nb in node_boxes)
            if not inside_node:
                final_detections.append(d)

    # Inject TAP (EQZ) / ADD CE-XX instances from vector search
    for eqz_cx, eqz_cy in eqz_instances:
        gx1 = int(eqz_cx - 35)
        gy1 = int(eqz_cy - 35)
        gx2 = int(eqz_cx + 35)
        gy2 = int(eqz_cy + 35)
        final_detections.append({
            "class_id": 19,
            "class_name": "TAP (EQZ) - ADD CE-XX",
            "confidence": 1.00,
            "bbox": (gx1, gy1, gx2, gy2),
            "center": (eqz_cx, eqz_cy)
        })

    # Draw annotated boxes
    annotated_img = img_bgr.copy()
    for d in final_detections:
        x1, y1, x2, y2 = d["bbox"]
        c_name = d["class_name"]
        conf_val = d["confidence"]

        color = (0, 255, 0) if "Tap" in c_name else (0, 0, 255)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
        label = f"{c_name} {conf_val:.2f}"
        cv2.putText(annotated_img, label, (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    out_file = output_dir / f"annotated_{input_path.stem}.jpg"
    cv2.imwrite(str(out_file), annotated_img)

    print("\n" + "=" * 50)
    print(f"🏆 SLIDING-WINDOW TILE DETECTION RESULTS")
    print(f"📄 Input File: {input_path.name}")
    print(f"🎯 Total Detected Symbols: {len(final_detections)}")
    print("=" * 50)

    class_counts = {}
    for d in final_detections:
        c_name = d["class_name"]
        class_counts[c_name] = class_counts.get(c_name, 0) + 1

    print(f"{'Symbol Class':<25} | {'Count':<10}")
    print("-" * 40)
    for c_name, count in sorted(class_counts.items()):
        print(f"{c_name:<25} | {count:<10}")
    print("=" * 50)
    print(f"\n🖼️ Full-Resolution Annotated Map Saved To:\n   [out_file] -> {out_file}\n")

if __name__ == "__main__":
    file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    test_model(file_arg)
