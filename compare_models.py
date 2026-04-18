"""
compare_models.py
=======================================================================
This code compared accuracy, speed, and size across all trained models.

Run this AFTER you have trained at least one model. It will:
  1. Locate the best checkpoint in each run directory.
  2. Evaluate each model on the validation split (mAP, precision, recall).
  3. Measure inference latency (mean ms, std ms, FPS) over multiple runs.
  4. Save a combined summary to results/model_comparison.csv.y
"""

import os
import time
import random
import pandas as pd
from ultralytics import YOLO
from PIL import Image

# ==================================== CONFIG ====================================
# IMPORTANT: Add or remove entries from RUN_DIRS to match the runs you trained.
# Each entry is the folder created inside runs/detect/ by the training scripts.

RUN_DIRS = [
    "runs/detect/YOLOv8n_flower_detection",
    "runs/detect/YOLOv8s_flower_detection",
    "runs/detect/RT-DETR-l_flower_detection",
]

DATA_YAML      = "dataset/data.yaml"   # same yaml used for training
IMG_SIZE       = 640    # adjust as needed
LATENCY_RUNS   = 20     # number of inference passes used to measure speed
MAX_VAL_IMAGES = 50     # images sampled for the latency benchmark
DEVICE         = "cpu"  # or gpu (0) if you've got one. 
                        # keep in mind that only DETR-l is GPU optimized.
OUTPUT_CSV     = "results/model_comparison.csv" # or whatever file name you want

# =================================================================================
def find_weights(run_dir: str) -> str | None:
    """Returns the path to best.pt (or last.pt) inside the run directories."""
    weights_dir = os.path.join(run_dir, "weights")
    if not os.path.isdir(weights_dir):
        return None
    for name in ("best.pt", "last.pt"):
        candidate = os.path.join(weights_dir, name)
        if os.path.isfile(candidate):
            return candidate
    # Fallback: any .pt file present
    pts = [f for f in os.listdir(weights_dir) if f.endswith(".pt")]
    return os.path.join(weights_dir, pts[0]) if pts else None

def load_val_images(data_yaml: str, max_images: int = MAX_VAL_IMAGES) -> list[str]:
    """Parses data.yaml and returns up to max_images validation image paths."""
    import yaml
    from pathlib import Path

    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)

    val_key = data.get("val") or data.get("validation") or data.get("test")
    val_path = (Path(data_yaml).parent / val_key).resolve()

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(val_path.rglob(ext))

    return [str(p) for p in images[:max_images]]

def measure_latency(
    model: YOLO,
    images: list[str],
    runs: int = LATENCY_RUNS,
) -> tuple[float, float, float]:
    """
    Returns
    -------
    mean_ms : float   average inference time in milliseconds
    std_ms  : float   standard deviation
    fps     : float   frames per second (1000 / mean_ms)
    """
    pil_images = [Image.open(p).convert("RGB") for p in images]

    # Warmup => excluded from timing
    for _ in range(3):
        model.predict(source=pil_images[0], imgsz=IMG_SIZE, device=DEVICE, verbose=False)

    times_ms = []
    for _ in range(runs):
        img = random.choice(pil_images)
        t0 = time.perf_counter()
        model.predict(source=img, imgsz=IMG_SIZE, device=DEVICE, verbose=False)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    mean_ms = sum(times_ms) / len(times_ms)
    std_ms  = (sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5
    fps     = 1000 / mean_ms

    return round(mean_ms, 2), round(std_ms, 2), round(fps, 2)

def main():
    print("=" * 65)
    print("Model Comparison - accuracy, speed, size")
    print("=" * 65)

    val_images = load_val_images(DATA_YAML)
    print(f"Found {len(val_images)} validation images for latency benchmark.\n")

    os.makedirs("results", exist_ok=True)
    rows = []

    for run_dir in RUN_DIRS:
        model_name = os.path.basename(run_dir)
        print(f"── {model_name}")

        weights = find_weights(run_dir)
        if not weights:
            print("Err: No weights found - skipping (have you trained this model yet?)\n")
            continue

        model = YOLO(weights)

        # ==================================== ACCURACY ====================================
        print("   Evaluating on validation set ...")
        metrics   = model.val(data=DATA_YAML, verbose=False)
        mAP50     = getattr(metrics.box, "map50", float("nan"))
        mAP50_95  = getattr(metrics.box, "map",   float("nan"))
        precision = getattr(metrics.box, "mp",    float("nan"))
        recall    = getattr(metrics.box, "mr",    float("nan"))

        # ==================================== MODEL SIZE ====================================
        size_mb = os.path.getsize(weights) / (1024 * 1024)
        try:
            params_m = sum(
                p.numel() for p in model.model.parameters() if p is not None
            ) / 1e6
        except Exception:
            params_m = float("nan")

        # ==================================== LATENCY ========================================
        print(f"Measuring latency ({LATENCY_RUNS} runs on {DEVICE}) ...")
        lat_mean, lat_std, fps = measure_latency(model, val_images)

        rows.append({
            "Model":            model_name,
            "Params (M)":       round(params_m, 2),
            "Size (MB)":        round(size_mb,  2),
            "mAP@0.5":          round(mAP50,    3),
            "mAP@0.5:0.95":     round(mAP50_95, 3),
            "Precision":        round(precision, 3),
            "Recall":           round(recall,    3),
            "Latency mean (ms)": lat_mean,
            "Latency std (ms)":  lat_std,
            "FPS":              fps,
        })
        print(f"mAP@0.5={mAP50:.3f} | FPS={fps}\n")

    if not rows:
        print("Err: No results collected - make sure at least one model has been trained!")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("=" * 65)
    print(df.to_string(index=False))
    print(f"\nFull comparison saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()