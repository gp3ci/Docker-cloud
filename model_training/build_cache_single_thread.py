"""
Pre-build Dataset Cache File (Single Threaded)
----------------------------------------------
Builds valid train.cache and val.cache matching Ultralytics cache schema & hash.
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics.data.utils import get_hash, save_dataset_cache_file
from ultralytics.data.dataset import DATASET_CACHE_VERSION

def prebuild_all_caches():
    base_dir = Path(__file__).parent
    dataset_dir = base_dir / "dataset"

    img_train = dataset_dir / "images" / "train"
    lbl_train = dataset_dir / "labels" / "train"

    img_val = dataset_dir / "images" / "val"
    lbl_val = dataset_dir / "labels" / "val"

    for img_d, lbl_d, name in [(img_train, lbl_train, "train"), (img_val, lbl_val, "val")]:
        img_files = sorted(list(img_d.glob("*.jpg")))
        lbl_files = [lbl_d / f"{p.stem}.txt" for p in img_files]

        img_paths_str = [str(p) for p in img_files]
        lbl_paths_str = [str(p) for p in lbl_files if p.exists()]

        hash_val = get_hash(lbl_paths_str + img_paths_str)
        labels_list = []

        nf, nm, ne, nc = 0, 0, 0, 0
        total = len(img_files)

        for img_path in img_files:
            lbl_path = lbl_d / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                nm += 1
                continue

            try:
                with Image.open(img_path) as im:
                    shape = (im.height, im.width)
            except Exception:
                nc += 1
                continue

            cls_labels = []
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_labels.append([int(parts[0])] + [float(x) for x in parts[1:5]])

            labels_arr = np.array(cls_labels, dtype=np.float32) if cls_labels else np.zeros((0, 5), dtype=np.float32)

            if len(labels_arr) == 0:
                ne += 1
            else:
                nf += 1

            labels_list.append({
                "im_file": str(img_path),
                "shape": shape,
                "cls": labels_arr[:, 0:1] if len(labels_arr) > 0 else np.zeros((0, 1), dtype=np.float32),
                "bboxes": labels_arr[:, 1:5] if len(labels_arr) > 0 else np.zeros((0, 4), dtype=np.float32),
                "segments": [],
                "keypoints": None,
                "normalized": True,
                "bbox_format": "xywh",
            })

        cache_dict = {
            "labels": labels_list,
            "hash": hash_val,
            "results": (nf, nm, ne, nc, total),
            "msgs": [],
            "version": DATASET_CACHE_VERSION,
        }

        # Save to all possible locations Ultralytics looks for
        paths = [
            dataset_dir / "labels" / f"{name}.cache",
            dataset_dir / "labels" / name / f"{name}.cache",
        ]
        for p in paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            save_dataset_cache_file("PreBuild", p, cache_dict, DATASET_CACHE_VERSION)
            print(f"✅ Saved cache to {p}")

if __name__ == "__main__":
    prebuild_all_caches()
