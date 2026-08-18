"""
YOLOv8 Model Training Script
----------------------------
Ported from AI-Quality_control_PROJECT reference pipeline.
Trains YOLOv8 with directional sensitivity controls (fliplr=0.0)
and early stopping on the prepared dataset in model_training/dataset/.
"""
import os
import torch
from pathlib import Path
from ultralytics import YOLO

def train_model():
    base_dir = Path(__file__).parent
    workspace_dir = base_dir.parent
    existing_best = workspace_dir / "backend" / "model_weights" / "best.pt"
    data_yaml_path = base_dir / "dataset" / "data.yaml"
    if not data_yaml_path.exists():
        data_yaml_path = base_dir / "data.yaml"

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}. Run prepare_dataset.py or convert_cvat_to_yolo.py first.")

    # Use existing best weights if available, otherwise start from yolov8n.pt
    if existing_best.exists():
        model_source = str(existing_best)
        print(f"🔄 Loading existing weights for fine-tuning: {model_source}")
    else:
        model_source = "yolov8n.pt"
        print(f"🚀 Initializing from baseline weights: {model_source}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"💻 Training Device: {device}")

    model = YOLO(model_source)

    # Train model using proven parameters from AI-Quality_control_PROJECT
    results = model.train(
        data=str(data_yaml_path),
        epochs=100,
        imgsz=640,            # Standard 640 or 1280 resolution
        batch=16,
        project=str(base_dir / "runs" / "detect"),
        name="symbol_detection_yolov8n_retrained",
        exist_ok=True,
        device=device,
        workers=0,            # Avoid Docker shared memory (/dev/shm) limits
        cache=False,          # Disable RAM cache to avoid shm semaphore limits
        patience=50,          # Early stopping after 50 epochs without improvement
        fliplr=0.0,           # CRITICAL: Disable horizontal flip to preserve symbol orientation
        save=True,
        plots=True,
    )

    metrics = model.val()
    print(f"📊 Final Validation mAP50-95: {metrics.box.map}")

    best_out = base_dir / "runs" / "detect" / "symbol_detection_yolov8n_retrained" / "weights" / "best.pt"
    print(f"\n✅ Training Complete! Best weights saved to:\n   {best_out}")
    if existing_best.parent.exists():
        print(f"To deploy to production backend, run:")
        print(f"Copy-Item '{best_out}' '{existing_best}'")

if __name__ == "__main__":
    train_model()
