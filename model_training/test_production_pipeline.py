"""
Test Production TelecomDetector Pipeline
----------------------------------------
Tests the full production backend detection pipeline in backend/app/services/vision.py
on any map PDF to verify actual production callouts & symbol detections.
"""
import sys
import os
from pathlib import Path
import fitz
import cv2
import numpy as np

# Add backend directory to sys.path
base_dir = Path(__file__).parent
backend_dir = base_dir.parent / "backend"
sys.path.insert(0, "/app")
sys.path.insert(0, str(backend_dir))

from app.services.vision import TelecomDetector

def test_production_detector(input_file=None):
    if not input_file:
        raw_samples = list((base_dir / "raw_samples").glob("*.pdf"))
        if raw_samples:
            input_file = raw_samples[0]
        else:
            print("❌ Please provide a PDF map path to test.")
            return

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return

    print(f"🚀 Initializing Production TelecomDetector...")
    main_path = "/app/model_weights/best.pt"
    ps_path = "/app/model_weights/power_supply_best.pt"
    node_path = "/app/model_weights/3x3_4x4_new_model.pt"
    internal_path = "/app/model_weights/Internal_best.pt"

    detector = TelecomDetector(main_path, ps_path, node_path, internal_path)

    print(f"📄 Processing Page 1 of PDF Map: {input_path.name}")
    doc = fitz.open(input_path)
    page = doc[0]
    dpi = 300
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1]

    print("🔍 Running Full Production Detector Pipeline...")
    detections = detector.detect_objects(img_bgr, conf_threshold=0.35)

    print("\n" + "=" * 50)
    print("🏆 PRODUCTION PIPELINE DETECTION RESULTS")
    print(f"📄 Input File: {input_path.name}")
    print(f"🎯 Total Detected Symbols: {len(detections)}")
    print("=" * 50)

    class_counts = {}
    for d in detections:
        c_name = d.get("cls", "unknown")
        class_counts[c_name] = class_counts.get(c_name, 0) + 1

    print(f"{'Symbol Class':<25} | {'Count':<10}")
    print("-" * 40)
    for c_name, count in sorted(class_counts.items()):
        print(f"{c_name:<25} | {count:<10}")
    print("=" * 50)

    # Draw annotated boxes
    annotated = img_bgr.copy()
    for d in detections:
        bbox = d.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        c_name = d.get("cls", "")
        conf_val = d.get("conf", 0.0)

        color = (0, 255, 0) if "tap" in c_name.lower() else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = f"{c_name} {conf_val:.2f}"
        cv2.putText(annotated, label, (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    out_file = base_dir / "test_outputs" / f"prod_annotated_{input_path.stem}.jpg"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), annotated)
    print(f"\n🖼️ Saved Production Pipeline Map to:\n   [out_file] -> {out_file}\n")

if __name__ == "__main__":
    file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    test_production_detector(file_arg)
