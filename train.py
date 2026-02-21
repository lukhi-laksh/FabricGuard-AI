"""
FabricGuard – Training Module  (GPU-STRICT)
============================================
Single-class (hole) YOLOv8m detection training.

This module ONLY runs on a dedicated NVIDIA CUDA GPU.
If no CUDA GPU is found the script raises RuntimeError immediately
before attempting any training — CPU fallback is intentionally disabled.

Data layout expected:
    data/
        hole_image/     <- fabric images (.jpg / .png / etc.)
        value_image/    <- YOLO label .txt files (same stem as images)

Label format (per line):
    0 x_center y_center width height   (all values 0.0–1.0, class 0 = hole)

Empty .txt = no hole in that image.
"""

# ── Force Windows to use the dedicated NVIDIA GPU (not Intel iGPU) ────────────
import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # GPU 0 = dedicated NVIDIA
_os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        # stable ordering across reboots

import os
import shutil
import random
import yaml
from pathlib import Path

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
HOLE_IMG_DIR = os.path.join(DATA_DIR, "hole_image")
VAL_LBL_DIR  = os.path.join(DATA_DIR, "value_image")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
YOLO_DATA    = os.path.join(BASE_DIR, "yolo_data")
YAML_PATH    = os.path.join(BASE_DIR, "dataset.yaml")
RUNS_DIR     = os.path.join(BASE_DIR, "runs")
MODEL_PATH   = os.path.join(MODELS_DIR, "hole_detector.pt")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _require_gpu() -> int:
    """
    Verify a CUDA GPU is available and return device index (0).
    Raises RuntimeError immediately if no GPU is found — CPU training
    is intentionally disabled for this project.
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError(
            "PyTorch is not installed.\n"
            "Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  ✗  NO CUDA GPU DETECTED — Training aborted!           \n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  Possible causes:\n"
            "  1. PyTorch CPU-only build installed.\n"
            "     Fix: pip install torch torchvision " \
            "--index-url https://download.pytorch.org/whl/cu121\n"
            "  2. NVIDIA drivers not installed / outdated.\n"
            "     Fix: install latest drivers from nvidia.com\n"
            "  3. CUDA_VISIBLE_DEVICES set to empty / -1.\n"
            "     Fix: unset CUDA_VISIBLE_DEVICES or set it to '0'\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    n   = torch.cuda.device_count()
    gpu = torch.cuda.get_device_properties(0)
    print(f"[GPU] Detected {n} CUDA GPU(s)")
    print(f"[GPU] Using → {gpu.name}  |  VRAM: {gpu.total_memory / 1e9:.1f} GB")
    print(f"[GPU] CUDA {torch.version.cuda}  |  cuDNN {torch.backends.cudnn.version()}")
    return 0   # always device 0


def _prepare_dataset(state: dict, val_split: float = 0.2) -> tuple[int, int]:
    """
    Scan data/hole_image/ and data/value_image/, match pairs, split
    80/20, and copy into yolo_data/train/ and yolo_data/val/.

    Returns (train_count, val_count).
    """
    _log(state, "Scanning dataset...")

    if not os.path.isdir(HOLE_IMG_DIR):
        raise FileNotFoundError(f"Images folder not found: {HOLE_IMG_DIR}")
    if not os.path.isdir(VAL_LBL_DIR):
        raise FileNotFoundError(f"Labels folder not found: {VAL_LBL_DIR}")

    all_images = sorted(
        f for f in os.listdir(HOLE_IMG_DIR)
        if f.lower().endswith(IMG_EXTS)
    )
    if not all_images:
        raise ValueError(f"No images found in {HOLE_IMG_DIR}")

    pairs = []
    skipped = []
    for img_fname in all_images:
        stem     = Path(img_fname).stem
        lbl_path = os.path.join(VAL_LBL_DIR, stem + ".txt")
        if os.path.exists(lbl_path):
            pairs.append((img_fname, stem + ".txt"))
        else:
            skipped.append(img_fname)

    if skipped:
        _log(state, f"  ⚠ Skipped {len(skipped)} images with no matching label")

    if not pairs:
        raise ValueError("No matched image-label pairs found. Add label .txt files.")

    _log(state, f"  ✓ {len(pairs)} matched pairs ready")

    # Shuffle and split
    random.seed(42)
    random.shuffle(pairs)
    sp          = max(1, int(len(pairs) * (1 - val_split)))
    train_pairs = pairs[:sp]
    val_pairs   = pairs[sp:]

    _log(state, f"  ✓ Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    # Rebuild yolo_data/
    if os.path.exists(YOLO_DATA):
        shutil.rmtree(YOLO_DATA)
    for sub in ("train/images", "train/labels", "val/images", "val/labels"):
        os.makedirs(os.path.join(YOLO_DATA, sub), exist_ok=True)

    for subset, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for img_fname, lbl_fname in split_pairs:
            shutil.copy2(
                os.path.join(HOLE_IMG_DIR, img_fname),
                os.path.join(YOLO_DATA, subset, "images", img_fname),
            )
            shutil.copy2(
                os.path.join(VAL_LBL_DIR, lbl_fname),
                os.path.join(YOLO_DATA, subset, "labels", lbl_fname),
            )

    _log(state, "  ✓ Dataset structure built")
    return len(train_pairs), len(val_pairs)


def _generate_yaml() -> str:
    """Write dataset.yaml for single-class hole detection."""
    cfg = {
        "path":  YOLO_DATA.replace("\\", "/"),
        "train": "train/images",
        "val":   "val/images",
        "nc":    1,
        "names": ["hole"],
    }
    with open(YAML_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return YAML_PATH


def _log(state: dict, msg: str):
    """Append message to state log (capped at 300 lines)."""
    state["log"].append(msg)
    if len(state["log"]) > 300:
        state["log"] = state["log"][-250:]


def train_model(epochs: int = 100, state: dict | None = None) -> bool:
    """
    Train YOLOv8m for hole detection.

    Args:
        epochs: Number of training epochs.
        state:  Shared dict updated in real-time for the UI.
                Keys: running, epoch, total_epochs, box_loss, cls_loss,
                      mAP50, precision, recall, done, error, log, history.

    Returns True on success, False on failure.
    """
    if state is None:
        state = {
            "running": True, "epoch": 0, "total_epochs": epochs,
            "box_loss": 0.0, "cls_loss": 0.0, "mAP50": 0.0,
            "precision": 0.0, "recall": 0.0,
            "done": False, "error": None, "log": [], "history": [],
        }

    try:
        from ultralytics import YOLO
    except ImportError:
        state["error"]   = "ultralytics not installed. Run: pip install ultralytics"
        state["running"] = False
        state["done"]    = True
        return False

    _log(state, "=" * 52)
    _log(state, "  FabricGuard  –  Hole Detection Training")
    _log(state, "=" * 52)

    # ── Dataset ──────────────────────────────────────────────
    try:
        _prepare_dataset(state)
    except Exception as e:
        state["error"]   = str(e)
        state["running"] = False
        state["done"]    = True
        return False

    yaml_path = _generate_yaml()
    _log(state, f"  ✓ dataset.yaml  nc=1  names=['hole']")

    # ── Device (GPU-strict) ──────────────────────────────────
    try:
        device = _require_gpu()      # raises RuntimeError if no GPU
    except RuntimeError as gpu_err:
        state["error"]   = str(gpu_err)
        state["running"] = False
        state["done"]    = True
        return False

    import torch
    gpu_name = torch.cuda.get_device_name(0)
    _log(state, f"  ✓ Device  : GPU 0 — {gpu_name}")
    _log(state, f"  ✓ VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    _log(state, f"  ✓ Model   : yolov8m.pt  (pretrained)")
    _log(state, f"  ✓ Epochs  : {epochs}  |  imgsz=640  |  optimizer=AdamW")
    _log(state, f"  ✓ Patience: 20 (early stopping)")
    _log(state, "=" * 52)

    # ── Load model ───────────────────────────────────────────
    model = YOLO("yolov8m.pt")

    # ── Callbacks ────────────────────────────────────────────
    def on_fit_epoch_end(trainer):
        """Called after each epoch (train + val complete)."""
        epoch = trainer.epoch + 1

        # Training losses from trainer.tloss
        try:
            tloss = trainer.tloss
            if tloss is not None and hasattr(tloss, "__len__") and len(tloss) >= 2:
                box_l = float(tloss[0])
                cls_l = float(tloss[1])
            elif tloss is not None:
                box_l = float(tloss)
                cls_l = 0.0
            else:
                box_l = cls_l = 0.0
        except Exception:
            box_l = cls_l = 0.0

        # Validation metrics
        metrics = getattr(trainer, "metrics", None) or {}
        map50   = float(metrics.get("metrics/mAP50(B)", 0.0))
        prec    = float(metrics.get("metrics/precision(B)", 0.0))
        rec     = float(metrics.get("metrics/recall(B)", 0.0))

        state["epoch"]     = epoch
        state["box_loss"]  = round(box_l, 4)
        state["cls_loss"]  = round(cls_l, 4)
        state["mAP50"]     = round(map50,  4)
        state["precision"] = round(prec,   4)
        state["recall"]    = round(rec,    4)
        state["history"].append({
            "epoch":    epoch,
            "box_loss": box_l,
            "cls_loss": cls_l,
            "mAP50":    map50,
        })

        msg = (f"Epoch {epoch:>4}/{epochs}  "
               f"│ box={box_l:.4f} │ cls={cls_l:.4f} │ mAP50={map50:.4f}")
        _log(state, msg)

    def on_train_end(trainer):
        _log(state, "=" * 52)
        _log(state, "  Training complete!")

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end",     on_train_end)

    # ── Train ────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    import torch
    torch.cuda.empty_cache()           # clear any leftover VRAM
    batch = -1                         # auto-batch (GPU always)

    results = model.train(
        data       = yaml_path,
        epochs     = epochs,
        imgsz      = 640,
        batch      = batch,
        optimizer  = "AdamW",
        patience   = 20,
        device     = device,
        project    = RUNS_DIR,
        name       = "hole_detection",
        exist_ok   = True,
        verbose    = False,
        save       = True,
        pretrained = True,
        plots      = False,
    )

    # ── Save best weights ────────────────────────────────────
    run_dir = os.path.join(RUNS_DIR, "hole_detection")
    best    = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(best):
        best = os.path.join(run_dir, "weights", "last.pt")

    if os.path.exists(best):
        shutil.copy2(best, MODEL_PATH)
        _log(state, f"  ✓ Model saved: {MODEL_PATH}")
    else:
        _log(state, "  ⚠ Could not locate best.pt")

    # ── Final metrics ────────────────────────────────────────
    try:
        rd    = results.results_dict
        final = float(rd.get("metrics/mAP50(B)", state["mAP50"]))
        state["mAP50"] = round(final, 4)
        _log(state, f"  ✓ Final mAP50  : {final:.4f}")
        _log(state, f"  ✓ Precision    : {state['precision']:.4f}")
        _log(state, f"  ✓ Recall       : {state['recall']:.4f}")
    except Exception:
        pass

    state["running"] = False
    state["done"]    = True
    return True


if __name__ == "__main__":
    import json
    s = {
        "running": True, "epoch": 0, "total_epochs": 100,
        "box_loss": 0.0, "cls_loss": 0.0, "mAP50": 0.0,
        "precision": 0.0, "recall": 0.0,
        "done": False, "error": None, "log": [], "history": [],
    }
    train_model(epochs=100, state=s)
    print("\n".join(s["log"]))
