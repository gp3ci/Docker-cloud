from ultralytics import YOLO

model = YOLO(r'c:/Users/lenovo/Desktop/PROJECTS/Internship-Gp3/symbol_detection_new/runs/detect/symbol_detection_yolov8n_v7/weights/best.pt')
metrics = model.val(data=r'c:/Users/lenovo/Desktop/PROJECTS/Internship-Gp3/symbol_detection_new/dataset/processed/data.yaml')

print("\n--- RESULTS ---")
print(f"Overall mAP50: {metrics.box.map50}")
for i, name in enumerate(metrics.names.values()):
    print(f"Class: {name:20s} Precision: {metrics.box.p[i]:.4f} Recall: {metrics.box.r[i]:.4f} mAP50: {metrics.box.maps[i]:.4f}")
print("---------------")
