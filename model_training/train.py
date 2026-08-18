"""
YOLOv8 Model Fine-Tuning Script with Anti-Overfitting Controls
---------------------------------------------------------------
Fine-tunes backend/model_weights/best.pt using transfer learning
on the balanced dataset in model_training/dataset/.
"""
import torch
from pathlib import Path
from ultralytics import YOLO

def main():
    base_dir = Path(__file__).parent.parent
    weights_path = base_dir / "backend" / "model_weights" / "best.pt"
    data_yaml_path = base_dir / "model_training" / "data.yaml"
    project_dir = base_dir / "model_training" / "runs"

    if not weights_path.exists():
        raise FileNotFoundError(f"Base weights not found at: {weights_path}")

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Data config not found at: {data_yaml_path}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if device == 0 else 'CPU'}")
    print(f"📦 Pre-trained base weights: {weights_path}")
    print(f"📊 Dataset config: {data_yaml_path}")

    # 1. Load baseline model weights
    model = YOLO(str(weights_path))

    # 2. Start fine-tuning with strong anti-overfitting controls
    results = model.train(
        data=str(data_yaml_path),
        epochs=100,             # Maximum epochs
        patience=15,            # Early stopping (halts training if val loss doesn't improve for 15 epochs)
        imgsz=1280,             # High resolution for small symbol detection
        batch=8,                # Adjust based on VRAM
        device=device,
        # ── Learning rate & Regularization ──
        lr0=0.001,              # Lower learning rate for fine-tuning pre-trained weights
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.001,     # Weight decay to prevent overfitting
        warmup_epochs=3.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # ── Anti-Overfitting Augmentations ──
        mosaic=1.0,             # Combines 4 random map tiles into 1 to prevent layout memorization
        degrees=10.0,           # Minor rotation jitter
        translate=0.1,          # Translation jitter
        scale=0.3,              # Multi-scale zooming (handles multi-DPI maps)
        fliplr=0.5,             # Horizontal flipping
        # ── Output Settings ──
        project=str(project_dir),
        name="best_retrained",
        save=True,
        plots=True,
        exist_ok=True,
    )

    best_trained_weights = project_dir / "best_retrained" / "weights" / "best.pt"
    print("\n✅ Training Complete!")
    print(f"New trained weights saved at: {best_trained_weights}")
    print(f"\nTo deploy to production, run:")
    print(f"Copy-Item '{best_trained_weights}' '{weights_path}'")

if __name__ == "__main__":
    main()
