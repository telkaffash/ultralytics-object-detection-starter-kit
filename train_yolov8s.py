"""
train_yolov8s.py
================
Train YOLOv8s (small) on your custom dataset.

YOLOv8s typically offers stronger accuracy-speed balance. It is a good default
choice for most detection tasks.

==================== How to use? =============================================

0. Make sure you've installed all dependencies (check README.md)
1. Update DATA_YAML to point to your dataset's data.yaml file.
2. Adjust hyperparameters in the CONFIG section if needed.
3. Run:
       python scripts/train_yolov8s.py

==================== Outputs =================================================

- Trained weights  : runs/detect/YOLOv8s_<project>/weights/best.pt
- Training plots   : runs/detect/YOLOv8s_<project>/
- Validation CSV   : results/YOLOv8s_metrics.csv
"""

import os
import pandas as pd
from ultralytics import YOLO

# ==================== CONFIG =================================================

DATA_YAML = "dataset/data.yaml"             # path to your dataset's data.yaml 
PROJECT_NAME = "YOLOv8s_flower_detection"   # folder name inside runs/detect/
EPOCHS       = 50                           # increase for better accuracy
IMG_SIZE     = 640                          # input resolution
BATCH        = 16                           # reduce if you run out of VRAM
DEVICE       ='cpu'                         # 0 = first GPU, "cpu" for CPU only

# ==============================================================================

def main():
    print("=" * 60)
    print("  Training YOLOv8s on custom dataset")
    print("=" * 60)
    
    # Load the pre-trained nano model (downloads automatically on first run)
    model = YOLO("yolov8s.pt")

    # Train
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        name=PROJECT_NAME,
        verbose=True,
    )

    # Evaluate on the validation split
    print("\nRunning validation on best checkpoint ...")
    metrics = model.val(data=DATA_YAML)

    # Save key metrics to CSV
    os.makedirs("results", exist_ok=True)
    results = [{
        "Model":         "YOLOv8s",
        "mAP@0.5":       round(metrics.box.map50, 3),
        "mAP@0.5:0.95":  round(metrics.box.map,   3),
        "Precision":     round(metrics.box.mp,     3),
        "Recall":        round(metrics.box.mr,     3),
    }]
    df = pd.DataFrame(results)
    df.to_csv("results/YOLOv8s_metrics.csv", index=False)

    print("\nTraining complete!")
    print(df.to_string(index=False))
    print(f"\nWeights saved to: runs/detect/{PROJECT_NAME}/weights/best.pt")
    print("Metrics saved to: results/YOLOv8s_metrics.csv")


if __name__ == "__main__":
    main()