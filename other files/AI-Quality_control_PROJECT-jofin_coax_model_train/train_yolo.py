from ultralytics import YOLO
import os

def train_model():
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (reverted to faster nano model)

    # Use absolute path to data.yaml
    data_path = r"c:/Users/lenovo/Desktop/PROJECTS/Internship-Gp3/symbol_detection_new/dataset/processed/data.yaml"

    # Train the model
    # epochs=100 is a good starting point, imgsz=640 is standard
    # batch size can be adjusted based on GPU memory
    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        batch=16, # Reverted to 16 since Nano model is smaller
        project=r"c:/Users/lenovo/Desktop/PROJECTS/Internship-Gp3/symbol_detection_new/runs/detect",
        name="symbol_detection_yolov8n_v7",
        exist_ok=True, # overwrite existing experiment
        device=0, # Use GPU 0 if available, else 'cpu'
        patience=50, # Stop early if no improvement for 50 epochs (accuracy/time improvement)
        fliplr=0.0   # Disable horizontal flip augmentation (prevents direction-sensitive symbols from being mirrored)
    )
    
    # Evaluate performance on validation set
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")

    # Export the model to ONNX format 
    # success = model.export(format="onnx")

if __name__ == "__main__":
    train_model()
