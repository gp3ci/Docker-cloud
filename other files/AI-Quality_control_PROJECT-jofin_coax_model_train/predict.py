from ultralytics import YOLO
import os
import cv2
from pathlib import Path

def predict_and_save():
    # Paths
    model_path = r"C:/Users/lenovo/Desktop/PROJECTS/Internship-Gp3/symbol_detection_new/runs/detect/symbol_detection_yolov8n_v3/weights/best.pt"
    source_dir = r"c:/Users/lenovo/Desktop/PROJECTS/Internship-Gp3/symbol_detection_new/before"
    output_dir = r"c:/Users/lenovo/Desktop/PROJECTS/Internship-Gp3/symbol_detection_new/inference_results"

    # Make output dir
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get list of images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in os.listdir(source_dir) if Path(f).suffix.lower() in image_extensions]
    
    if not images:
        print(f"No images found in {source_dir}")
        return

    print(f"Found {len(images)} images. Running inference...")

    # Run inference
    for img_file in images:
        img_path = os.path.join(source_dir, img_file)
        
        # Predict
        results = model(img_path)
        
        # Visualize and save
        for result in results:
            # Plot the results on the image (returns a numpy array)
            plotted_img = result.plot()
            
            # Save the image
            save_path = os.path.join(output_dir, img_file)
            cv2.imwrite(save_path, plotted_img)
            # print(f"Saved: {save_path}")

    print(f"Inference complete. Results saved to {output_dir}")

if __name__ == "__main__":
    predict_and_save()
