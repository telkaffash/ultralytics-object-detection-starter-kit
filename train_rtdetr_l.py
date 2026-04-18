"""
train_rtdetr_l.py
==============================================================================
Train RT-DETR-l (Real-Time Detection Transformer, large)

RT-DETR is a transformer-based detector that achieves high accuracy while
maintaining real-time inference speed, thus the name. Note that transformers
are optimized to run on GPUs.

==================== How to use? =============================================

0. Make sure you've installed all dependencies (check README.md)
1. Update DATA_YAML to point to your dataset's data.yaml file.
2. Tune hyperparameters in the CONFIG section as needed.
3. Run:
       python scripts/train_rtdetr_l.py

Note: RT-DETR benefits from more VRAM than YOLOv8. If you hit OOM errors, reduce 
      BATCH or switch to DEVICE = "cpu" (much slower).
      
      In this case, you may want to upload your workspace to Google Colab and run
      on their free GPU, otherwise if you're locally training on cpu I'd be praying 
      for you.

==================== Outputs =================================================

- Trained weights  : runs/detect/RT-DETR-l_<project>/weights/best.pt
- Training plots   : runs/detect/RT-DETR-l_<project>/
- Validation CSV   : results/RT-DETR-l_metrics.csv
"""

import os
import pandas as pd
from ultralytics import YOLO   # ultralytics wraps RT-DETR via YOLO interface

# ==================== CONFIG =================================================

DATA_YAML    = "dataset/data.yaml"

PROJECT_NAME = "RT-DETR-l_flower_detection"
EPOCHS       = 50
IMG_SIZE     = 640
BATCH        = 8       # RT-DETR-l is heavier; lower batch if needed
DEVICE       ='cpu'       # 0 = first GPU, "cpu" for CPU only

# ==============================================================================

def main():
    print("=" * 60)
    print("  Training RT-DETR-l on custom dataset")
    print("=" * 60)

    # Load RT-DETR-l via the ultralytics YOLO interface
    # (downloads pre-trained weights automatically on first run)
    model = YOLO("rtdetr-l.pt")

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        name=PROJECT_NAME,
        verbose=True,
    )

    print("\nRunning validation on best checkpoint ...")
    metrics = model.val(data=DATA_YAML)

    os.makedirs("results", exist_ok=True)
    results = [{
        "Model":         "RT-DETR-l",
        "mAP@0.5":       round(metrics.box.map50, 3),
        "mAP@0.5:0.95":  round(metrics.box.map,   3),
        "Precision":     round(metrics.box.mp,     3),
        "Recall":        round(metrics.box.mr,     3),
    }]
    df = pd.DataFrame(results)
    df.to_csv("results/RT-DETR-l_metrics.csv", index=False)

    print("\nTraining complete!")
    print(df.to_string(index=False))
    print(f"\nWeights saved to: runs/detect/{PROJECT_NAME}/weights/best.pt")
    print("Metrics saved to: results/RT-DETR-l_metrics.csv")


if __name__ == "__main__":
    main()