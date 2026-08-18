# 🎯 Telecom Vision Model Retraining & Dataset Pipeline

This folder contains the complete setup for preparing map datasets and fine-tuning the YOLO object detection model (`best.pt`).

---

## 📂 Directory Layout & Upload Folder

```text
Docker-cloud-main/
└── model_training/
    ├── raw_samples/          <-- 📥 UPLOAD YOUR RAW MAP PDFs OR CVAT ZIP EXPORTS HERE
    ├── prepare_dataset.py     <-- Automatically slices maps & splits train/val datasets
    ├── train.py              <-- PyTorch / YOLOv8 fine-tuning script (with anti-overfitting)
    ├── data.yaml             <-- Class mapping (A, ADD CE-XX, E, G, H, J, etc.)
    ├── dataset/              <-- (Gitignored) Store training/val images and labels here
    │   ├── images/
    │   │   ├── train/
    │   │   └── val/
    │   └── labels/
    │       ├── train/
    │       └── val/
    └── runs/                 # (Gitignored) Training checkpoints & validation graphs
```

---

## 🚀 How to Use

### Step 1: Upload Your Files
Drop your **sample PDF maps**, **image tiles**, or **CVAT export `.zip` files** directly into:
👉 `model_training/raw_samples/`

### Step 2: Prepare Dataset
Run the automated preparation script:
```bash
python model_training/prepare_dataset.py
```
* Automatically converts PDF maps to 600 DPI 1280x1280 image tiles.
* Unpacks CVAT ZIP exports and moves image/label pairs into an 80% train / 20% validation split.

### Step 3: Run Fine-Tuning
Execute fine-tuning:
```bash
python model_training/train.py
```
* Pre-loaded with anti-overfitting augmentations (`mosaic=1.0`, `scale=0.3`, `patience=15`).

### Step 4: Deploy Weights
Copy the output weights to production:
```powershell
Copy-Item "model_training/runs/best_retrained/weights/best.pt" "backend/model_weights/best.pt"
```
