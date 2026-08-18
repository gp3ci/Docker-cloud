"""
CVAT XML to YOLO Dataset Converter
-----------------------------------
Ported from AI-Quality_control_PROJECT reference pipeline.
Reads raw CVAT XML exports from model_training/raw_samples/ or model_training/dataset/
and converts bounding boxes to standard YOLO format with 80/20 train/val split.
"""
import os
import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path

def convert_cvat_to_yolo(base_dir: Path, output_dir: Path, train_ratio: float = 0.8):
    """
    Converts CVAT XML annotations to YOLO format and organizes into train/val split.
    """
    annotation_files = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".xml") or f == "annotations.xml":
                annotation_files.append(os.path.join(root, f))

    if not annotation_files:
        print(f"⚠️ No annotation XML files found in {base_dir}!")
        return

    print(f"📦 Found {len(annotation_files)} XML annotation files.")
    classes = set()
    for xml_file in annotation_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for label in root.findall(".//label"):
                name_elem = label.find("name")
                if name_elem is not None and name_elem.text:
                    classes.add(name_elem.text.strip())
            # Also check direct box labels
            for box in root.findall(".//box"):
                lbl = box.get("label")
                if lbl:
                    classes.add(lbl.strip())
        except Exception as e:
            print(f"Error reading {xml_file}: {e}")

    if not classes:
        print("⚠️ No labels found in XML files!")
        return

    class_list = sorted(list(classes))
    class_map = {name: i for i, name in enumerate(class_list)}
    print(f"🏷️ Class Mapping ({len(class_list)} classes): {class_map}")

    # Prepare output directories
    images_train_dir = output_dir / "images" / "train"
    images_val_dir   = output_dir / "images" / "val"
    labels_train_dir = output_dir / "labels" / "train"
    labels_val_dir   = output_dir / "labels" / "val"

    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_images = []

    for xml_file in annotation_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except Exception:
            continue
        
        xml_dir = os.path.dirname(xml_file)

        for image_tag in root.findall(".//image"):
            image_name = image_tag.get("name")
            width = float(image_tag.get("width", 0))
            height = float(image_tag.get("height", 0))
            if width == 0 or height == 0:
                continue

            full_image_path = os.path.join(xml_dir, image_name)
            if not os.path.exists(full_image_path):
                basename = os.path.basename(image_name)
                found = False
                for r, d, f in os.walk(xml_dir):
                    if basename in f:
                        full_image_path = os.path.join(r, basename)
                        found = True
                        break
                if not found:
                    continue

            label_lines = []
            for box in image_tag.findall("box"):
                label = box.get("label")
                if not label or label not in class_map:
                    continue

                xtl = float(box.get("xtl"))
                ytl = float(box.get("ytl"))
                xbr = float(box.get("xbr"))
                ybr = float(box.get("ybr"))

                dw = 1.0 / width
                dh = 1.0 / height

                w = xbr - xtl
                h = ybr - ytl
                x_center = xtl + w / 2.0
                y_center = ytl + h / 2.0

                x_center *= dw
                y_center *= dh
                w *= dw
                h *= dh

                class_id = class_map[label]
                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            all_images.append((full_image_path, label_lines))

    if not all_images:
        print("⚠️ No valid images matched with XML annotations.")
        return

    random.seed(42)
    random.shuffle(all_images)
    split_index = int(len(all_images) * train_ratio)
    train_data = all_images[:split_index]
    val_data   = all_images[split_index:]

    def save_data(data, img_dir, lbl_dir):
        for img_path, labels in data:
            unique_name = f"{Path(img_path).parent.name}_{Path(img_path).name}"
            dest_img_path = img_dir / unique_name
            shutil.copy2(img_path, dest_img_path)

            label_filename = unique_name.rsplit('.', 1)[0] + ".txt"
            dest_label_path = lbl_dir / label_filename
            with open(dest_label_path, "w") as f:
                f.write("\n".join(labels))

    print(f"💾 Saving {len(train_data)} training samples...")
    save_data(train_data, images_train_dir, labels_train_dir)

    print(f"💾 Saving {len(val_data)} validation samples...")
    save_data(val_data, images_val_dir, labels_val_dir)

    yaml_content = f"""# Auto-generated dataset config
train: {os.path.abspath(images_train_dir)}
val: {os.path.abspath(images_val_dir)}

nc: {len(class_list)}
names: {class_list}
"""
    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_content)

    print("✅ CVAT XML to YOLO conversion complete!")
    print(f"📄 Config written to: {output_dir / 'data.yaml'}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "raw_samples"
    dataset_dir = base_dir / "dataset"
    convert_cvat_to_yolo(raw_dir, dataset_dir)
