"""
================================================================================
  FabricGuard AI — YOLOv8 Hole Detection Trainer (Google Colab)
  Author  : FabricGuard AI Team
  Version : 3.0.0
  Target  : Google Colab Pro (GPU + High-RAM runtime)
  Run     : !python online.py
================================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import sys
import shutil
import random
import zipfile
import logging
import yaml
import glob
from pathlib import Path

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s » %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("FabricGuard")

DIVIDER = "=" * 72


def banner(title: str) -> None:
    log.info(DIVIDER)
    log.info(f"  {title}")
    log.info(DIVIDER)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 0 — Environment Check
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 0 · Environment & Dependency Check")

# --- Install ultralytics if not present (idempotent) -------------------------
try:
    import ultralytics  # noqa: F401
    log.info("ultralytics already installed ✓")
except ImportError:
    log.info("Installing ultralytics …")
    os.system("pip install -q ultralytics")

# --- GPU availability ---------------------------------------------------------
try:
    import torch

    if not torch.cuda.is_available():
        log.warning(
            "⚠  No GPU detected! Training will fall back to CPU which is VERY slow.\n"
            "   → Go to Runtime ▸ Change runtime type ▸ GPU and re-run."
        )
        DEVICE = "cpu"
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU detected ✓  →  {gpu_name}  ({vram_gb:.1f} GB VRAM)")
        DEVICE = "0"
except Exception as exc:
    log.error(f"PyTorch import failed: {exc}")
    sys.exit(1)

# ── Google Drive Mount ────────────────────────────────────────────────────────
banner("STEP 1 · Mount Google Drive")

try:
    from google.colab import drive, files  # type: ignore
    drive.mount("/content/drive", force_remount=False)
    log.info("Google Drive mounted at /content/drive ✓")
    IN_COLAB = True
except ImportError:
    log.warning("Not running inside Colab — skipping Drive mount.")
    IN_COLAB = False

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
DRIVE_IMAGE_DIR = Path("/content/drive/MyDrive/mmm/image")
DRIVE_LABEL_DIR = Path("/content/drive/MyDrive/mmm/lable")   # matches user's folder name

DATASET_ROOT    = Path("/content/dataset")
YAML_PATH       = Path("/content/hole_dataset.yaml")
RUN_NAME        = "hole_detector_v1"
RUNS_DIR        = Path("/content/runs/detect")
EXPORT_DIR      = Path("/content/drive/MyDrive/mmm/trained_model")

VAL_SPLIT       = 0.20  # 80 / 20 train-val split
RANDOM_SEED     = 42

TRAIN_CFG = dict(
    model      = "yolov8n.pt",   # nano → fast Colab training; swap to yolov8s/m for accuracy
    epochs     = 100,
    imgsz      = 640,
    batch      = -1,             # auto-batch based on GPU VRAM
    device     = DEVICE,
    workers    = 8,
    cache      = True,
    pretrained = True,
    optimizer  = "AdamW",
    project    = str(RUNS_DIR),
    name       = RUN_NAME,
    exist_ok   = True,
    patience   = 30,             # early stopping
    save       = True,
    plots      = True,
    verbose    = True,
)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Source Validation
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 2 · Validate Source Data in Google Drive")

if not DRIVE_IMAGE_DIR.exists():
    log.error(f"Image directory not found: {DRIVE_IMAGE_DIR}")
    log.error("Check your Drive path and re-run.")
    sys.exit(1)

if not DRIVE_LABEL_DIR.exists():
    log.error(f"Label directory not found: {DRIVE_LABEL_DIR}")
    log.error("Check your Drive path and re-run.")
    sys.exit(1)

raw_images = sorted(
    [p for p in DRIVE_IMAGE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
)
raw_labels = sorted(
    [p for p in DRIVE_LABEL_DIR.iterdir() if p.suffix.lower() == ".txt"]
)

log.info(f"Total images found : {len(raw_images)}")
log.info(f"Total labels found : {len(raw_labels)}")

if len(raw_images) == 0:
    log.error("No images found!  Supported formats: .jpg, .jpeg, .png")
    sys.exit(1)

if len(raw_labels) == 0:
    log.error("No label (.txt) files found in the label directory!")
    sys.exit(1)

# Build a stem → path map for labels
label_map = {lbl.stem: lbl for lbl in raw_labels}

# ── Pair images ↔ labels (skip un-labelled images) ───────────────────────────
paired, skipped_no_label = [], []

for img in raw_images:
    if img.stem in label_map:
        paired.append((img, label_map[img.stem]))
    else:
        skipped_no_label.append(img.name)

if skipped_no_label:
    log.warning(
        f"Skipped {len(skipped_no_label)} image(s) with no matching label:\n"
        + "\n".join(f"   • {n}" for n in skipped_no_label[:20])
        + ("\n   … (truncated)" if len(skipped_no_label) > 20 else "")
    )

if len(paired) == 0:
    log.error(
        "No image–label pairs found!\n"
        "Ensure image filenames match label filenames (without extension)."
    )
    sys.exit(1)

log.info(f"Valid image-label pairs : {len(paired)}")

# ── Class Index Sanity Check (spot-check first 50 labels) ────────────────────
log.info("Validating class indices (expected: 0 only) …")
bad_class_files = []
for _, lbl_path in paired[:50]:
    with open(lbl_path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                bad_class_files.append(f"{lbl_path.name}:{line_no} (bad format)")
                break
            cls_id = int(parts[0])
            if cls_id != 0:
                bad_class_files.append(
                    f"{lbl_path.name}:{line_no} (class_id={cls_id}, expected 0)"
                )
                break

if bad_class_files:
    log.error(
        "Class index mismatch detected in label files:\n"
        + "\n".join(f"   ✗ {e}" for e in bad_class_files)
    )
    sys.exit(1)

log.info("Class indices look correct ✓")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Build YOLO Dataset Structure
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 3 · Build YOLO Dataset (train / val split)")

# Clean previous dataset
if DATASET_ROOT.exists():
    shutil.rmtree(DATASET_ROOT)
    log.info("Removed stale dataset directory.")

for split in ("train", "val"):
    (DATASET_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)

# Shuffle & split
random.seed(RANDOM_SEED)
random.shuffle(paired)
split_idx   = int(len(paired) * (1 - VAL_SPLIT))
train_pairs = paired[:split_idx]
val_pairs   = paired[split_idx:]

log.info(f"Train samples : {len(train_pairs)}")
log.info(f"Val   samples : {len(val_pairs)}")


def copy_pairs(pairs, split: str) -> None:
    img_dst = DATASET_ROOT / split / "images"
    lbl_dst = DATASET_ROOT / split / "labels"
    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, img_dst / img_path.name)
        shutil.copy2(lbl_path, lbl_dst / lbl_path.name)


log.info("Copying training data …")
copy_pairs(train_pairs, "train")
log.info("Copying validation data …")
copy_pairs(val_pairs, "val")
log.info("Dataset structure built ✓")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Write dataset.yaml
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 4 · Write YOLO Dataset YAML")

dataset_cfg = {
    "path"  : str(DATASET_ROOT),
    "train" : "train/images",
    "val"   : "val/images",
    "nc"    : 1,
    "names" : ["hole"],
}

with open(YAML_PATH, "w") as f:
    yaml.dump(dataset_cfg, f, default_flow_style=False, sort_keys=False)

log.info(f"Dataset YAML written → {YAML_PATH}")
log.info(f"  nc    : {dataset_cfg['nc']}")
log.info(f"  names : {dataset_cfg['names']}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Memory Optimisation Tips
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 5 · Memory Optimisation")

import gc
gc.collect()
if DEVICE != "cpu":
    torch.cuda.empty_cache()
    log.info(
        f"GPU memory cleared. "
        f"Available: {torch.cuda.memory_reserved(0)/1e9:.2f} GB reserved."
    )

# If batch=-1 is not supported by the installed version, fallback gracefully
if DEVICE == "cpu":
    TRAIN_CFG["batch"] = 8
    TRAIN_CFG["workers"] = 0
    TRAIN_CFG["cache"] = False
    log.warning("CPU mode: batch=8, workers=0, cache=False applied.")

log.info("Memory settings configured ✓")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Train YOLOv8
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 6 · Start YOLOv8 Training")

from ultralytics import YOLO  # noqa: E402 — installed above

log.info(f"Base model   : {TRAIN_CFG['model']}")
log.info(f"Epochs       : {TRAIN_CFG['epochs']}")
log.info(f"Image size   : {TRAIN_CFG['imgsz']}")
log.info(f"Batch        : {TRAIN_CFG['batch']} (auto)" if TRAIN_CFG["batch"] == -1 else f"Batch: {TRAIN_CFG['batch']}")
log.info(f"Device       : {TRAIN_CFG['device']}")
log.info(f"Optimizer    : {TRAIN_CFG['optimizer']}")
log.info(f"Early stop   : patience={TRAIN_CFG['patience']} epochs")

model = YOLO(TRAIN_CFG.pop("model"))

results = model.train(
    data    = str(YAML_PATH),
    **TRAIN_CFG,
)

log.info("Training complete ✓")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Locate best.pt
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 7 · Locate best.pt")

WEIGHTS_CANDIDATES = list(
    (RUNS_DIR / RUN_NAME / "weights").glob("best.pt")
)
if not WEIGHTS_CANDIDATES:
    # Fallback: search entire /content/runs tree
    WEIGHTS_CANDIDATES = list(Path("/content/runs").rglob("best.pt"))

if not WEIGHTS_CANDIDATES:
    log.error(
        "best.pt not found after training!\n"
        "Check training logs above for errors."
    )
    sys.exit(1)

BEST_PT = WEIGHTS_CANDIDATES[0]
log.info(f"best.pt located → {BEST_PT}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — Export Model (ONNX + TorchScript)
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 8 · Export Model to ONNX & TorchScript")

trained_model = YOLO(str(BEST_PT))

log.info("Exporting to ONNX …")
try:
    trained_model.export(
        format   = "onnx",
        imgsz    = 640,
        simplify = True,
        opset    = 17,
        dynamic  = False,
    )
    log.info("ONNX export done ✓")
except Exception as exc:
    log.warning(f"ONNX export skipped: {exc}")

log.info("Exporting to TorchScript …")
try:
    trained_model.export(format="torchscript", imgsz=640)
    log.info("TorchScript export done ✓")
except Exception as exc:
    log.warning(f"TorchScript export skipped: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9 — Save Artifacts to Google Drive
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 9 · Save Artifacts to Google Drive")

if IN_COLAB:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy best.pt
    dst_best = EXPORT_DIR / "best.pt"
    shutil.copy2(BEST_PT, dst_best)
    log.info(f"best.pt copied to Drive → {dst_best}")

    # Copy ONNX if exists
    onnx_src = BEST_PT.parent / "best.onnx"
    if onnx_src.exists():
        shutil.copy2(onnx_src, EXPORT_DIR / "best.onnx")
        log.info(f"best.onnx copied to Drive ✓")

    # Copy TorchScript if exists
    ts_src = BEST_PT.parent / "best.torchscript"
    if ts_src.exists():
        shutil.copy2(ts_src, EXPORT_DIR / "best.torchscript")
        log.info(f"best.torchscript copied to Drive ✓")

    # Copy training plots / metrics
    run_dir = RUNS_DIR / RUN_NAME
    for artifact in ["results.png", "confusion_matrix.png", "results.csv"]:
        src = run_dir / artifact
        if src.exists():
            shutil.copy2(src, EXPORT_DIR / artifact)

    log.info("All artifacts saved to Google Drive ✓")
else:
    log.info("Not in Colab — Drive export skipped.")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 10 — Zip the Full Run Folder
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 10 · Zip Trained Model Folder")

run_dir    = RUNS_DIR / RUN_NAME
zip_path   = Path(f"/content/{RUN_NAME}.zip")

log.info(f"Zipping {run_dir} …")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for file in run_dir.rglob("*"):
        if file.is_file():
            zf.write(file, file.relative_to(run_dir.parent))

log.info(f"Zip created → {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")

# Copy zip to Drive too
if IN_COLAB:
    shutil.copy2(zip_path, EXPORT_DIR / zip_path.name)
    log.info(f"Zip also saved to Drive ✓")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 11 — Auto Download via Colab
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 11 · Trigger Auto-Download")

if IN_COLAB:
    log.info("Downloading best.pt …")
    files.download(str(BEST_PT))

    log.info(f"Downloading {zip_path.name} …")
    files.download(str(zip_path))
else:
    log.info("Not in Colab — manual download required.")
    log.info(f"Artifacts located at: {BEST_PT.parent}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 12 — Validation Run (Quick)
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 12 · Post-Training Validation")

val_results = trained_model.val(
    data   = str(YAML_PATH),
    imgsz  = 640,
    device = DEVICE,
    split  = "val",
    plots  = True,
    save_json = False,
)

log.info("Validation Results:")
log.info(f"  mAP@50       : {val_results.box.map50:.4f}")
log.info(f"  mAP@50-95    : {val_results.box.map:.4f}")
log.info(f"  Precision    : {val_results.box.mp:.4f}")
log.info(f"  Recall       : {val_results.box.mr:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
#  DONE
# ══════════════════════════════════════════════════════════════════════════════
banner("✅  ALL STEPS COMPLETE")

log.info(f"Trained weights  → {BEST_PT}")
log.info(f"Drive export dir → {EXPORT_DIR}")
log.info(f"Zip archive      → {zip_path}")
log.info("")
log.info("To run inference:")
log.info("  from ultralytics import YOLO")
log.info("  model = YOLO('best.pt')")
log.info("  results = model.predict('your_image.jpg', conf=0.25)")
log.info("  results[0].show()")
log.info(DIVIDER)
