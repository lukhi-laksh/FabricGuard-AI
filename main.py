"""
FabricGuard – Main Application  (GPU-STRICT)
==============================================
Single-class hole detection system.

Run:
    python main.py

UI:  http://127.0.0.1:8000
"""

# ── GPU preference (use GPU if present, CPU otherwise) ──────────────────────
import os
# Allow GPU 0 if available; never forcibly hide the GPU.
if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "-1":
    del os.environ["CUDA_VISIBLE_DEVICES"]   # un-block GPU if someone set -1
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── Eager import of detect so GPU is resolved NOW at startup ──────────────────
# (lazy import inside route handlers caused the uvicorn CUDA timing bug)
import detect as _detect_module  # noqa: F401 — triggers module-level GPU check

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
HOLE_IMG_DIR = os.path.join(DATA_DIR, "hole_image")
VAL_LBL_DIR  = os.path.join(DATA_DIR, "value_image")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
UPLOADS_DIR  = os.path.join(BASE_DIR, "uploads")

for _d in [HOLE_IMG_DIR, VAL_LBL_DIR, MODELS_DIR,
           os.path.join(UPLOADS_DIR, "images"),
           os.path.join(UPLOADS_DIR, "annotated"),
           os.path.join(UPLOADS_DIR, "videos"),
           os.path.join(UPLOADS_DIR, "detected_frames")]:
    os.makedirs(_d, exist_ok=True)

# ── App setup ───────────────────────────────────────────────────────────────────
app = FastAPI(title="FabricGuard – Hole Detection")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ── Shared training state (updated by background thread) ────────────────────────
training_state: dict = {
    "running":      False,
    "epoch":        0,
    "total_epochs": 100,
    "box_loss":     0.0,
    "cls_loss":     0.0,
    "mAP50":        0.0,
    "precision":    0.0,
    "recall":       0.0,
    "done":         False,
    "error":        None,
    "log":          [],
    "history":      [],
}

# ── Shared webcam stats (updated by streaming generator each frame) ──────────────
webcam_state: dict = {
    "hole_count":    0,
    "max_confidence": 0.0,
    "running":       False,
}


# ── Routes: UI ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ── Routes: System ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    model_ready = os.path.exists(os.path.join(MODELS_DIR, "hole_detector.pt"))
    return {"status": "healthy", "model_ready": model_ready}


@app.get("/dataset-info")
async def dataset_info():
    """Return current numbers for images and labels in data/."""
    images = [f for f in os.listdir(HOLE_IMG_DIR) if f.lower().endswith(IMG_EXTS)] \
             if os.path.isdir(HOLE_IMG_DIR) else []
    labels = [f for f in os.listdir(VAL_LBL_DIR)  if f.endswith(".txt")] \
             if os.path.isdir(VAL_LBL_DIR)  else []

    img_stems = {Path(f).stem for f in images}
    lbl_stems = {Path(f).stem for f in labels}
    matched   = img_stems & lbl_stems

    return {
        "total_images":   len(images),
        "total_labels":   len(labels),
        "matched_pairs":  len(matched),
        "missing_labels": sorted(img_stems - lbl_stems),
        "missing_images": sorted(lbl_stems - img_stems),
        "ready":          len(matched) > 0 and len(img_stems - lbl_stems) == 0,
    }


# ── Routes: Dataset Upload ──────────────────────────────────────────────────────
@app.post("/upload-images")
async def upload_images(files: list[UploadFile] = File(...)):
    """Save fabric images to data/hole_image/."""
    saved = []
    for f in files:
        if not any(f.filename.lower().endswith(ext) for ext in IMG_EXTS):
            continue
        dest = os.path.join(HOLE_IMG_DIR, f.filename)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(f.filename)
    return {"saved": len(saved), "files": saved}


@app.post("/upload-labels")
async def upload_labels(files: list[UploadFile] = File(...)):
    """Save YOLO label .txt files to data/value_image/ and verify matching."""
    saved = []
    for f in files:
        if not f.filename.endswith(".txt"):
            continue
        dest = os.path.join(VAL_LBL_DIR, f.filename)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(f.filename)

    # Recompute matching after upload
    info = await dataset_info()
    return {
        "saved":   len(saved),
        "files":   saved,
        **info,
    }


@app.post("/clear-dataset")
async def clear_dataset():
    """Remove all uploaded images and labels (start fresh)."""
    for d in [HOLE_IMG_DIR, VAL_LBL_DIR]:
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
    return {"message": "Dataset cleared"}


# ── Routes: Training ──────────────────────────────────────────────────────────
@app.post("/train")
async def start_training(epochs: int = Form(100)):
    """Start YOLOv8m training in a background thread."""
    global training_state

    if training_state.get("running"):
        return JSONResponse(status_code=409,
                            content={"message": "Training already running"})

    training_state = {
        "running":      True,
        "epoch":        0,
        "total_epochs": epochs,
        "box_loss":     0.0,
        "cls_loss":     0.0,
        "mAP50":        0.0,
        "precision":    0.0,
        "recall":       0.0,
        "done":         False,
        "error":        None,
        "log":          [],
        "history":      [],
    }

    def _run():
        try:
            from train import train_model
            train_model(epochs=epochs, state=training_state)
        except Exception as e:
            training_state["error"]   = str(e)
            training_state["running"] = False
            training_state["done"]    = True

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"message": "Training started", "epochs": epochs}


@app.get("/train-status")
async def train_status():
    """Poll for live training progress."""
    return training_state


# ── Routes: Detection ─────────────────────────────────────────────────────────
@app.post("/detect-image")
async def detect_image_endpoint(file: UploadFile = File(...)):
    """Detect holes in a single uploaded image."""
    from detect import detect_image

    img_path = os.path.join(UPLOADS_DIR, "images", file.filename)
    with open(img_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = detect_image(img_path)
    return result


@app.post("/detect-video")
async def detect_video_endpoint(file: UploadFile = File(...)):
    """Detect holes in an uploaded video (returns first detected frame)."""
    from detect import detect_video

    vid_path = os.path.join(UPLOADS_DIR, "videos", file.filename)
    with open(vid_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = detect_video(vid_path)
    return result


@app.get("/webcam-devices")
def list_webcam_devices():
    """Enumerate all connected cameras and return their indices + labels."""
    import cv2 as _cv2
    devices = []
    # Probe indices 0-9 quickly
    for idx in range(10):
        cap = _cv2.VideoCapture(idx, _cv2.CAP_DSHOW)   # CAP_DSHOW = fast on Windows
        if cap.isOpened():
            # Try to get a backend name if available
            backend = cap.getBackendName() if hasattr(cap, "getBackendName") else "Camera"
            label = f"Camera {idx}" if idx == 0 else f"External Camera {idx}"
            devices.append({"index": idx, "label": label})
            cap.release()
    return {"devices": devices}


@app.get("/webcam-stats")
def webcam_stats():
    """Current hole count & confidence from the live webcam stream."""
    return webcam_state


@app.get("/webcam-feed")
def webcam_feed(camera_index: int = 0):
    """MJPEG stream with live hole detection overlay. Accepts ?camera_index=N"""
    from detect import generate_webcam_frames
    webcam_state["running"] = True
    webcam_state["hole_count"] = 0
    webcam_state["max_confidence"] = 0.0

    def _stream():
        try:
            for chunk in generate_webcam_frames(camera_index=camera_index, state=webcam_state):
                yield chunk
        finally:
            webcam_state["running"] = False
            webcam_state["hole_count"] = 0
            webcam_state["max_confidence"] = 0.0

    return StreamingResponse(
        _stream(),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


# ── Entry ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Device detection at startup ──────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            _p        = torch.cuda.get_device_properties(0)
            device_line = f"Device → GPU  {_p.name}  ({_p.total_memory/1e9:.1f} GB VRAM)  ✓"
        else:
            device_line = "Device → CPU  (no CUDA GPU — running on CPU)"
    except Exception as _e:
        device_line = f"Device → CPU  (PyTorch error: {_e})"

    print()
    print("=" * 58)
    print("   FabricGuard  –  Hole Detection System")
    print("=" * 58)
    print(f"   {device_line}")
    print(f"   UI  →  http://127.0.0.1:8000")
    print(f"   Data images  →  data/hole_image/")
    print(f"   Data labels  →  data/value_image/")
    print("=" * 58)
    print()

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
