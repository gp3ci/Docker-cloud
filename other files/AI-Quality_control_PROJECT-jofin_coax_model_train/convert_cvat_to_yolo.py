import os
import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path

def convert_cvat_to_yolo(base_dir, output_dir, train_ratio=0.8):
    """
    Converts CVAT XML annotations to YOLO format and organizes into train/val split.
    """
    
    # 1. Identify all annotation files and collect all unique class names
    annotation_files = []
    for root, dirs, files in os.walk(base_dir):
        if "annotations.xml" in files:
            annotation_files.append(os.path.join(root, "annotations.xml"))

    if not annotation_files:
        print("No annotations.xml files found!")
        return

    classes = set()
    for xml_file in annotation_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for label in root.findall(".//label"):
             name = label.find("name").text
             classes.add(name)
    
    # Sort classes for consistent ID mapping
    class_list = sorted(list(classes))
    class_map = {name: i for i, name in enumerate(class_list)}
    print(f"Found classes: {class_map}")

    # Prepare output directories
    images_train_dir = Path(output_dir) / "images" / "train"
    images_val_dir = Path(output_dir) / "images" / "val"
    labels_train_dir = Path(output_dir) / "labels" / "train"
    labels_val_dir = Path(output_dir) / "labels" / "val"

    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Process each annotation file
    all_images = [] # Collect (image_path, label_content) tuples

    for xml_file in annotation_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # The 'before' images are relative to the XML file's parent directory
        xml_dir = os.path.dirname(xml_file)
        
        for image_tag in root.findall("image"):
            image_name = image_tag.get("name")
            width = float(image_tag.get("width"))
            height = float(image_tag.get("height"))
            
            # Construct full path to the image
            # image_name in XML might be "before/Pilot_..."
            # xml file is in "dataset/Breacayb_pilot_..."
            # so full path is "dataset/Breacayb_pilot_.../before/Pilot_..."
            # verifying if image_name includes the subdirectory or not
            
            full_image_path = os.path.join(xml_dir, image_name)
            
            # Check if file exists, sometimes paths in XML might be slightly off
            if not os.path.exists(full_image_path):
                 # Try to find it if it's just the filename
                 basename = os.path.basename(image_name)
                 found = False
                 for r, d, f in os.walk(xml_dir):
                     if basename in f:
                         full_image_path = os.path.join(r, basename)
                         found = True
                         break
                 if not found:
                     print(f"Warning: Image {image_name} not found, skipping.")
                     continue

            label_lines = []
            for box in image_tag.findall("box"):
                label = box.get("label")
                if label not in class_map:
                    continue
                
                xtl = float(box.get("xtl"))
                ytl = float(box.get("ytl"))
                xbr = float(box.get("xbr"))
                ybr = float(box.get("ybr"))
                
                # Convert to YOLO (normalized center_x, center_y, w, h)
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
            
            # We add valid images even if they have no labels (empty label file)
            # But usually for object detection we prefer images with labels or specific negatives.
            # Here we include them.
            all_images.append((full_image_path, label_lines))

    # 3. Split and Save
    random.shuffle(all_images)
    split_index = int(len(all_images) * train_ratio)
    train_data = all_images[:split_index]
    val_data = all_images[split_index:]

    def save_data(data, img_dir, lbl_dir):
        for img_path, labels in data:
            # Copy image
            unique_name = f"{Path(img_path).parent.parent.name}_{Path(img_path).name}"
            # Ensure unique name to avoid collisions from different folders
            # e.g. Pilot_... from folder 1 and Pilot_... from folder 2 might have same name?
            # actually they seem to have unique IDs in filenames, but let's be safe.
            # folder struct: dataset/Breacayb_pilot_4131451/before/Pilot_4131451_1.jpg
            # parent.parent.name is "Breacayb_pilot_4131451"
            
            dest_img_path = img_dir / unique_name
            shutil.copy2(img_path, dest_img_path)
            
            # Create label file
            label_filename = unique_name.rsplit('.', 1)[0] + ".txt"
            dest_label_path = lbl_dir / label_filename
            
            with open(dest_label_path, "w") as f:
                f.write("\n".join(labels))

    print(f"Saving {len(train_data)} training images...")
    save_data(train_data, images_train_dir, labels_train_dir)
    
    print(f"Saving {len(val_data)} validation images...")
    save_data(val_data, images_val_dir, labels_val_dir)

    # 4. Create data.yaml
    yaml_content = f"""
train: {os.path.abspath(images_train_dir)}
val: {os.path.abspath(images_val_dir)}

nc: {len(class_list)}
names: {class_list}
"""
    with open(Path(output_dir) / "data.yaml", "w") as f:
        f.write(yaml_content)

    print("Conversion complete!")
    print(f"Data.yaml created at {Path(output_dir) / 'data.yaml'}")

if __name__ == "__main__":
    base_dataset_dir = r"c:/Users/lenovo/Desktop/PROJECTS/Internship-Gp3/symbol_detection_new/dataset"
    output_dataset_dir = r"c:/Users/lenovo/Desktop/PROJECTS/Internship-Gp3/symbol_detection_new/dataset/processed"
    
    convert_cvat_to_yolo(base_dataset_dir, output_dataset_dir)
