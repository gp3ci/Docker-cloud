import json
from ultralytics import YOLO

def main():
    model = YOLO(r'c:/Users/lenovo/Desktop/PROJECTS/Internship-Gp3/symbol_detection_new/runs/detect/symbol_detection_yolov8n_v7/weights/best.pt')
    metrics = model.val(data=r'c:/Users/lenovo/Desktop/PROJECTS/Internship-Gp3/symbol_detection_new/dataset/processed/data.yaml')

    results = {
        "overall": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "classes": {}
    }

    for i, class_idx in enumerate(metrics.ap_class_index):
        name = metrics.names[class_idx]
        results["classes"][name] = {
            "precision": float(metrics.box.p[i]),
            "recall": float(metrics.box.r[i]),
            "mAP50": float(metrics.box.maps[i])
        }

    with open("v7_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    print("METRICS EXTRACTED SUCCESSFULLY")

if __name__ == '__main__':
    main()
