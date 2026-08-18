"""
Visualize Labeled Symbols on Image Tiles
---------------------------------------
Draws bounding boxes & class labels from model_training/dataset/
and saves sample annotated preview images into model_training/dataset/preview_samples/
"""
import cv2
import os
import random
from pathlib import Path

# Class names matching 22-class dataset config
CLASS_NAMES = [
    "1x4 Node", "2way_splitter", "2x2 Node", "3Way_splitter", "3_way_amplifier",
    "3x3 Node", "4x4 Node", "Block", "Booster", "Dual_Amplifier", "Equalizer",
    "Int_2way_splitter", "Line_Extender", "Power_Block", "Splice", "Splitter",
    "Splitter_DC", "Splitter_int_DC", "Tag_id", "Taps", "Terminator", "power_supply"
]

def visualize_dataset_samples(num_samples=10):
    base_dir = Path(__file__).parent
    img_dir = base_dir / "dataset" / "images" / "train"
    lbl_dir = base_dir / "dataset" / "labels" / "train"
    out_dir = base_dir / "dataset" / "preview_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = list(img_dir.glob("*.jpg"))
    if not img_files:
        print("No image tiles found in dataset/images/train/")
        return

    random.seed(42)
    selected_imgs = random.sample(img_files, min(num_samples, len(img_files)))

    for img_path in selected_imgs:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        box_count = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])

            x1 = int((xc - bw / 2.0) * w)
            y1 = int((yc - bh / 2.0) * h)
            x2 = int((xc + bw / 2.0) * w)
            y2 = int((yc + bh / 2.0) * h)

            cls_label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class_{cls_id}"

            # Green box for Taps/Amplifiers/Splitters, Red for CEX/Nodes
            color = (0, 0, 255) if cls_id == 19 else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, cls_label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            box_count += 1

        if box_count > 0:
            out_file = out_dir / f"preview_{img_path.name}"
            cv2.imwrite(str(out_file), img)
            print(f"🖼️ Saved preview with {box_count} symbols: {out_file.name}")

if __name__ == "__main__":
    visualize_dataset_samples(10)
