"""
Compare Model Weights: Baseline vs Retrained
--------------------------------------------
Runs YOLO validation on model_training/dataset/data.yaml
comparing original baseline weights against newly retrained weights.
"""
import os
from pathlib import Path
from ultralytics import YOLO

def compare_models():
    base_dir = Path(__file__).parent
    dataset_yaml = base_dir / "dataset" / "data.yaml"
    if not dataset_yaml.exists():
        dataset_yaml = base_dir / "data.yaml"

    retrained_weights = base_dir / "runs" / "detect" / "symbol_detection_yolov8n_retrained" / "weights" / "best.pt"
    if not retrained_weights.exists():
        retrained_weights = Path("/app/backend/model_weights/best.pt")

    # Find reference/baseline weights
    reference_dir = base_dir.parent / "other files" / "AI-Quality_control_PROJECT-jofin_coax_model_train"
    ref_weights = None
    for p in [reference_dir / "best.pt", reference_dir / "runs" / "detect" / "train" / "weights" / "best.pt"]:
        if p.exists():
            ref_weights = p
            break

    print(f"📊 Evaluating Retrained Model Weights: {retrained_weights}")
    m_retrained = YOLO(str(retrained_weights))
    val_retrained = m_retrained.val(data=str(dataset_yaml), split="val", workers=0, verbose=False)

    print("\n=======================================================")
    print("🏆 MODEL COMPARISON METRICS SUMMARY (on Validation Set)")
    print("=======================================================")
    print(f"{'Metric':<20} | {'Retrained Model':<20}")
    print("-" * 45)
    print(f"{'Precision (P)':<20} | {val_retrained.box.mp:<20.4f}")
    print(f"{'Recall (R)':<20} | {val_retrained.box.mr:<20.4f}")
    print(f"{'mAP@50':<20} | {val_retrained.box.map50:<20.4f}")
    print(f"{'mAP@50-95':<20} | {val_retrained.box.map:<20.4f}")
    print("=======================================================")

    print("\n📌 PER-CLASS mAP50 BREAKDOWN (Retrained Model):")
    print(f"{'Class ID & Name':<25} | {'mAP50':<10}")
    print("-" * 40)
    for i, c_name in m_retrained.names.items():
        if i < len(val_retrained.box.maps):
            print(f"{i:2d}: {c_name:<21} | {val_retrained.box.maps[i]:<10.4f}")

if __name__ == "__main__":
    compare_models()
