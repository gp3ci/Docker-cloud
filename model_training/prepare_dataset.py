"""
Automated Dataset Preparation Script for Telecom Vision Training
------------------------------------------------------------------
This script processes raw uploads in `model_training/raw_samples/`:
1. Extract ZIP archives (CVAT / Roboflow exports).
2. Convert PDF maps to 600 DPI image tiles (1280x1280).
3. Split tiles and labels 80/20 into train/val datasets.
"""
import os
import shutil
import zipfile
import random
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image

def extract_zips(raw_dir: Path):
    """Extracts any zip files in raw_samples (e.g. CVAT exports)."""
    for zip_file in raw_dir.glob("*.zip"):
        print(f"📦 Extracting archive: {zip_file.name}")
        extract_to = raw_dir / zip_file.stem
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted to {extract_to}")

def slice_image_to_tiles(img_path: Path, tile_size: int = 1280, overlap: int = 256) -> list[Image.Image]:
    """Slices a large map image into overlapping 1280x1280 tiles."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    tiles = []

    step = tile_size - overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            box = (x, y, min(x + tile_size, w), min(y + tile_size, h))
            tile = img.crop(box)
            # Pad tile if smaller than tile_size
            if tile.size != (tile_size, tile_size):
                padded = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
                padded.paste(tile, (0, 0))
                tile = padded
            tiles.append((tile, x, y))
    return tiles

def convert_pdf_to_tiles(pdf_path: Path, output_img_dir: Path, dpi: int = 600):
    """Converts a PDF map into 1280x1280 image tiles."""
    doc = fitz.open(pdf_path)
    print(f"📄 Processing PDF: {pdf_path.name} ({len(doc)} pages) at {dpi} DPI")

    for pno in range(len(doc)):
        page = doc[pno]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Save temporary full page image
        temp_img_path = output_img_dir / f"{pdf_path.stem}_page{pno+1}.png"
        pix.save(str(temp_img_path))

        # Slice into tiles
        tiles = slice_image_to_tiles(temp_img_path)
        temp_img_path.unlink() # remove temp full image

        for idx, (tile, x, y) in enumerate(tiles):
            tile_filename = f"{pdf_path.stem}_p{pno+1}_tile_{idx}_{x}_{y}.jpg"
            tile.save(output_img_dir / tile_filename, quality=95)
        print(f"   Generated {len(tiles)} tiles for page {pno+1}")

def prepare_dataset():
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "raw_samples"
    dataset_dir = base_dir / "dataset"

    # Define target directories
    img_train = dataset_dir / "images" / "train"
    img_val   = dataset_dir / "images" / "val"
    lbl_train = dataset_dir / "labels" / "train"
    lbl_val   = dataset_dir / "labels" / "val"

    for d in [img_train, img_val, lbl_train, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract ZIPs
    extract_zips(raw_dir)

    # Step 2: Search for direct image + label pairs (from CVAT / Roboflow)
    all_images = list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.png"))
    all_labels = list(raw_dir.rglob("*.txt"))

    # Step 3: Process PDF maps if any uploaded
    pdf_files = list(raw_dir.glob("*.pdf"))
    if pdf_files:
        for pdf_path in pdf_files:
            convert_pdf_to_tiles(pdf_path, img_train)

    # Step 4: Organize image + label pairs with 80/20 train/val split
    image_label_pairs = []
    label_dict = {lbl.stem: lbl for lbl in all_labels if lbl.name != ".gitkeep"}

    for img in all_images:
        if img.stem in label_dict:
            image_label_pairs.append((img, label_dict[img.stem]))

    if image_label_pairs:
        print(f"🔍 Found {len(image_label_pairs)} annotated image-label pairs.")
        random.seed(42)
        random.shuffle(image_label_pairs)

        split_idx = int(len(image_label_pairs) * 0.8)
        train_pairs = image_label_pairs[:split_idx]
        val_pairs   = image_label_pairs[split_idx:]

        for img, lbl in train_pairs:
            shutil.copy(img, img_train / img.name)
            shutil.copy(lbl, lbl_train / lbl.name)

        for img, lbl in val_pairs:
            shutil.copy(img, img_val / img.name)
            shutil.copy(lbl, lbl_val / lbl.name)

        print(f"✅ Dataset Organized: {len(train_pairs)} train pairs, {len(val_pairs)} val pairs.")
    else:
        print("ℹ️ No annotated image/label pairs found yet in raw_samples.")
        print("   Upload your PDF maps or CVAT ZIP exports to `model_training/raw_samples/` and run this script again.")

if __name__ == "__main__":
    prepare_dataset()
