"""
Automated Pseudo-Labeler for Uploaded PDF Maps
-----------------------------------------------
1. Converts PDF map Page 1 in raw_samples/ to 300 DPI 1280x1280 image tiles.
2. Runs best.pt to auto-detect symbols (amplifier, tap, splitter, LE, etc.).
3. Suppresses symbols inside Node boxes (only Node remains).
4. Squares with B or S are classified as Class 14 (Splice), NOT Taps.
5. Small solid black triangles and solid black rectangles are EXCLUDED from Terminators.
6. Equalizers (Class 10) are kept distinct from Terminators (Class 20).
"""
import os
import re
import shutil
import random
from pathlib import Path
import fitz
import cv2
import numpy as np
from ultralytics import YOLO

NODE_CLASSES = {0, 2, 5, 6}  # 1x4 Node, 2x2 Node, 3x3 Node, 4x4 Node

def auto_label_maps():
    base_dir = Path(__file__).parent
    raw_samples_dir = base_dir / "raw_samples"
    dataset_dir = base_dir / "dataset"

    img_train = dataset_dir / "images" / "train"
    img_val   = dataset_dir / "images" / "val"
    lbl_train = dataset_dir / "labels" / "train"
    lbl_val   = dataset_dir / "labels" / "val"

    for d in [img_train, img_val, lbl_train, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    weights_path = Path("/app/model_weights/best.pt")
    if not weights_path.exists():
        weights_path = base_dir.parent / "backend" / "model_weights" / "best.pt"
    if not weights_path.exists():
        weights_path = base_dir / "backend" / "model_weights" / "best.pt"

    if not weights_path.exists():
        print(f"❌ Base weights file not found at {weights_path}")
        return

    print(f"🤖 Loading pre-trained YOLO model from: {weights_path}")
    model = YOLO(str(weights_path))

    pdf_files = [p for p in raw_samples_dir.glob("*.pdf") if "Details" not in p.name]
    if not pdf_files:
        print(f"⚠️ No map PDF files found in {raw_samples_dir}")
        return

    print(f"📄 Found {len(pdf_files)} PDF map files to auto-label.")
    all_samples = []

    for pdf_path in pdf_files:
        print(f"🔍 Auto-labeling PDF: {pdf_path.name}")
        doc = fitz.open(pdf_path)
        dpi = 300
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        # Process ONLY Page 1 (main circuit map sheet)
        for pno in range(min(1, len(doc))):
            page = doc[pno]
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1]
            h, w = img_bgr.shape[:2]

            tile_size = 1280
            overlap = 256
            step = tile_size - overlap

            # PyMuPDF text search for Splice Blocks [B] and [S]
            splice_instances = []
            for word_info in page.get_text("words"):
                w_text = word_info[4].strip()
                if w_text in ["B", "S", "[B]", "[S]", "SPLICE"]:
                    r_x0, r_y0, r_x1, r_y1 = word_info[:4]
                    cx_px = (r_x0 + r_x1) / 2.0 * zoom
                    cy_px = (r_y0 + r_y1) / 2.0 * zoom
                    splice_instances.append((cx_px, cy_px))

            # PyMuPDF text search for Tap numbers (e.g. 4, 7, 8, 10, 11, 12, 14, 17, 20, 23, 26, 29, 32)
            tap_val_instances = []
            for word_info in page.get_text("words"):
                w_text = word_info[4].strip()
                if re.match(r"^\(?([478]|10|11|12|14|17|20|23|26|29|32)\)?$", w_text):
                    r_x0, r_y0, r_x1, r_y1 = word_info[:4]
                    cx_px = (r_x0 + r_x1) / 2.0 * zoom
                    cy_px = (r_y0 + r_y1) / 2.0 * zoom
                    tap_val_instances.append((cx_px, cy_px))

            # PyMuPDF text search for EQZ / CE text (CEX Taps)
            eqz_text_instances = []
            for search_str in ["EQZ", "CE-", "CE ", "CE_"]:
                text_rects = page.search_for(search_str)
                for r in text_rects:
                    cx_px = (r.x0 + r.x1) / 2.0 * zoom
                    cy_px = (r.y0 + r.y1) / 2.0 * zoom
                    eqz_text_instances.append((cx_px, cy_px))

            tile_count = 0
            for y in range(0, h, step):
                for x in range(0, w, step):
                    tile = img_bgr[y:min(y+tile_size, h), x:min(x+tile_size, w)]
                    th, tw = tile.shape[:2]

                    if th < tile_size or tw < tile_size:
                        padded = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
                        padded[:th, :tw] = tile
                        tile = padded

                    results = model(tile, conf=0.20, verbose=False)[0]
                    raw_boxes = []

                    if hasattr(results, "boxes") and results.boxes is not None:
                        ALLOWED_CLASSES = {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21}

                        for box in results.boxes:
                            cls_id = int(box.cls[0])
                            conf_val = float(box.conf[0])

                            if cls_id not in ALLOWED_CLASSES:
                                continue

                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bw_px = x2 - x1
                            bh_px = y2 - y1

                            # House/Pole Number Filter: Small round shapes (<42px) are house/pole numbers!
                            if cls_id == 19 and (bw_px < 42 or bh_px < 42) and conf_val < 0.35:
                                continue

                            # EXCLUDE solid black triangles and solid black rectangles from Terminators (Class 20)
                            cx_int = int((x1 + x2) / 2.0)
                            cy_int = int((y1 + y2) / 2.0)
                            crop = tile[max(0, cy_int-15):min(tile_size, cy_int+15), max(0, cx_int-15):min(tile_size, cx_int+15)]
                            if crop.size > 0:
                                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                # Solid black fill has low mean intensity < 70
                                if cls_id == 20 and np.mean(gray_crop) < 70.0:
                                    continue  # Filter out solid black triangles and black rectangles!

                            raw_boxes.append({
                                "cls_id": cls_id,
                                "conf": conf_val,
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "xc": (x1 + x2) / 2.0, "yc": (y1 + y2) / 2.0,
                                "bw": bw_px, "bh": bh_px,
                            })

                    # RULE 1: Node Containment Suppression (Only Node stays; symbols inside Node are removed!)
                    node_boxes = [b for b in raw_boxes if b["cls_id"] in NODE_CLASSES]
                    filtered_boxes = []

                    for b in raw_boxes:
                        if b["cls_id"] in NODE_CLASSES:
                            filtered_boxes.append(b)
                        else:
                            # Check if symbol center is inside any Node box
                            inside_node = False
                            for nb in node_boxes:
                                if nb["x1"] <= b["xc"] <= nb["x2"] and nb["y1"] <= b["yc"] <= nb["y2"]:
                                    inside_node = True
                                    break
                            if not inside_node:
                                filtered_boxes.append(b)

                    label_lines = []
                    for b in filtered_boxes:
                        w_box = b["bw"] / tile_size
                        h_box = b["bh"] / tile_size
                        xc_box = b["xc"] / tile_size
                        yc_box = b["yc"] / tile_size
                        label_lines.append(f"{b['cls_id']} {xc_box:.6f} {yc_box:.6f} {w_box:.6f} {h_box:.6f}")

                    # RULE 2: Inject Splice Blocks [B] and [S] (Class 14 Splice, NOT Taps)
                    for sp_cx, sp_cy in splice_instances:
                        if x <= sp_cx <= x + tile_size and y <= sp_cy <= y + tile_size:
                            local_cx = (sp_cx - x) / tile_size
                            local_cy = (sp_cy - y) / tile_size
                            box_size = 50.0 / tile_size
                            # Remove any Tap label overlapping this Splice location
                            label_lines = [l for l in label_lines if not (l.startswith("19 ") and abs(float(l.split()[1]) - local_cx) < 0.03 and abs(float(l.split()[2]) - local_cy) < 0.03)]
                            # Add Class 14 (Splice)
                            if not any(abs(float(l.split()[1]) - local_cx) < 0.03 and abs(float(l.split()[2]) - local_cy) < 0.03 for l in label_lines if l.startswith("14 ")):
                                label_lines.append(f"14 {local_cx:.6f} {local_cy:.6f} {box_size:.6f} {box_size:.6f}")

                    # RULE 3: Inject Vector Tap Instances (Class 19 Taps) from PDF text coordinates
                    for tap_cx, tap_cy in tap_val_instances:
                        if x <= tap_cx <= x + tile_size and y <= tap_cy <= y + tile_size:
                            local_cx = (tap_cx - x) / tile_size
                            local_cy = (tap_cy - y) / tile_size
                            # Don't inject if inside Node box or Splice location
                            inside_node = any(nb["x1"] <= (tap_cx - x) <= nb["x2"] and nb["y1"] <= (tap_cy - y) <= nb["y2"] for nb in node_boxes)
                            is_splice = any(abs(sp_cx - tap_cx) < 30 and abs(sp_cy - tap_cy) < 30 for sp_cx, sp_cy in splice_instances)
                            if not inside_node and not is_splice:
                                box_size = 55.0 / tile_size
                                if not any(abs(float(l.split()[1]) - local_cx) < 0.03 and abs(float(l.split()[2]) - local_cy) < 0.03 for l in label_lines if l.startswith("19 ")):
                                    label_lines.append(f"19 {local_cx:.6f} {local_cy:.6f} {box_size:.6f} {box_size:.6f}")

                    if label_lines:
                        sample_name = f"{pdf_path.stem}_p{pno+1}_tile_{y}_{x}"
                        tile_img_path = raw_samples_dir / f"{sample_name}.jpg"
                        cv2.imwrite(str(tile_img_path), tile)
                        all_samples.append((tile_img_path, label_lines))
                        tile_count += 1

            print(f"   Page 1: Auto-generated {tile_count} labeled image tiles.")

    if not all_samples:
        print("⚠️ No labeled tiles generated.")
        return

    random.seed(42)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    val_samples   = all_samples[split_idx:]

    def save_split(samples, img_dir, lbl_dir):
        for img_path, lines in samples:
            shutil.move(str(img_path), str(img_dir / img_path.name))
            lbl_name = img_path.stem + ".txt"
            with open(lbl_dir / lbl_name, "w") as f:
                f.write("\n".join(lines))

    save_split(train_samples, img_train, lbl_train)
    save_split(val_samples, img_val, lbl_val)

    print(f"\n✅ Auto-labeling Complete!")
    print(f"📊 Dataset Stats: {len(train_samples)} Train tiles, {len(val_samples)} Validation tiles.")

if __name__ == "__main__":
    auto_label_maps()
