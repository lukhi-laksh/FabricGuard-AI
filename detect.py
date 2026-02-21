"""
FabricGuard – Detection Module  (GPU-STRICT)
==============================================
Single-class (hole) YOLOv8 detection for images, video, and webcam.
All inference runs exclusively on the dedicated NVIDIA GPU.

Public API:
    detect_image(image_path, threshold=0.4)
        -> dict {hole_count, max_confidence, detection_status, detections, annotated_url, message}

    detect_video(video_path, threshold=0.4, frame_skip=5)
        -> dict {detection_status, hole_count, max_confidence, timestamp, frame_url, message}

    generate_webcam_frames(threshold=0.4, camera_index=0)
        -> generator yielding MJPEG bytes
"""

# ── Force dedicated NVIDIA GPU BEFORE torch loads ───────────────────────
import os as _os
# Only override if the shell hasn't pinned another value
if _os.environ.get("CUDA_VISIBLE_DEVICES", "0") in ("", "-1"):
    _os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # dedicated NVIDIA only
elif "CUDA_VISIBLE_DEVICES" not in _os.environ:
    _os.environ["CUDA_VISIBLE_DEVICES"] = "0"
_os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # stable index across reboots

import os
import cv2
import numpy as np
from pathlib import Path

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "hole_detector.pt")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
ANN_DIR     = os.path.join(UPLOADS_DIR, "annotated")
FRAMES_DIR  = os.path.join(UPLOADS_DIR, "detected_frames")

THRESHOLD = 0.4

os.makedirs(ANN_DIR,    exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# ── Resolve device at import time ─────────────────────────────────────────────
def _resolve_device():
    """
    Detect the best available device.
    Returns 'cuda:0' when a CUDA GPU is present, otherwise 'cpu'.
    Never raises — always falls back to CPU.
    """
    try:
        import torch as _t
        if _t.cuda.is_available():
            name    = _t.cuda.get_device_name(0)
            vram_gb = _t.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[FabricGuard] Device → GPU  {name} ({vram_gb:.1f} GB VRAM)  ✓")
            return "cuda:0"
        else:
            print("[FabricGuard] Device → CPU  (no CUDA GPU found — running on CPU)")
            return "cpu"
    except ImportError:
        print("[FabricGuard] Device → CPU  (PyTorch not installed yet)")
        return "cpu"

_DEVICE = _resolve_device()


def _get_device() -> str:
    """
    Return the best available device string ('cuda:0' or 'cpu').
    Re-checks on every call in case the environment changed (rare but safe).
    """
    global _DEVICE
    try:
        import torch as _t
        _DEVICE = "cuda:0" if _t.cuda.is_available() else "cpu"
    except Exception:
        _DEVICE = "cpu"
    return _DEVICE


def _load_model():
    """Load the trained hole detection model. Raises FileNotFoundError if missing."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Please train the model first.\n"
            f"Expected: {MODEL_PATH}"
        )
    from ultralytics import YOLO
    return YOLO(MODEL_PATH)


def _read_image(image_path: str) -> np.ndarray:
    """Read image via OpenCV, fallback to PIL for exotic formats."""
    img = cv2.imread(image_path)
    if img is None:
        from PIL import Image as _PIL
        img = cv2.cvtColor(np.array(_PIL.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return img


def _draw_boxes(image: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw red bounding boxes + confidence labels on an OpenCV image.
    Returns the annotated image (modifies in-place).
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf  = det["confidence"]
        label = f"Hole {conf:.1%}"
        color = (0, 0, 255)   # BGR → Red

        # Box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        lbl_y1 = max(y1 - th - 8, 0)
        cv2.rectangle(image, (x1, lbl_y1), (x1 + tw + 6, y1), color, -1)

        # Label text (white on red)
        cv2.putText(image, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image


def _run_inference(model, source, device: str, threshold: float) -> list:
    """
    Run model inference and return list of detections.
    Each detection: {bbox: [x1,y1,x2,y2], confidence: float}
    """
    results = model(source, device=device, conf=threshold, verbose=False)
    boxes   = results[0].boxes
    dets    = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            dets.append({
                "bbox":       [round(v) for v in box.xyxy[0].tolist()],
                "confidence": round(float(box.conf.item()), 4),
            })

    return dets


# ── Image Detection ────────────────────────────────────────────────────────────
def detect_image(image_path: str, threshold: float = THRESHOLD) -> dict:
    """
    Detect holes in a single image.

    Returns:
        hole_count        : int   – total holes found
        max_confidence    : float – highest confidence among detections
        detection_status  : bool  – True if ≥1 hole found
        detections        : list  – [{bbox, confidence}, ...]
        annotated_url     : str | None – URL of image with red boxes drawn
        message           : str   – human-readable result
        error             : bool
    """
    try:
        model  = _load_model()
        device = _get_device()
    except FileNotFoundError as e:
        return {
            "hole_count": 0, "max_confidence": 0.0,
            "detection_status": False, "detections": [],
            "annotated_url": None, "message": str(e), "error": True,
        }
    except Exception as e:
        return {
            "hole_count": 0, "max_confidence": 0.0,
            "detection_status": False, "detections": [],
            "annotated_url": None, "message": f"Load error: {e}", "error": True,
        }

    try:
        dets = _run_inference(model, image_path, device, threshold)
    except Exception as e:
        return {
            "hole_count": 0, "max_confidence": 0.0,
            "detection_status": False, "detections": [],
            "annotated_url": None, "message": f"Inference error: {e}", "error": True,
        }

    hole_count = len(dets)
    max_conf   = round(max((d["confidence"] for d in dets), default=0.0), 4)
    status     = hole_count > 0

    # Build message
    if status:
        if hole_count == 1:
            message = f"Hole detected ({max_conf:.1%})"
        else:
            message = f"{hole_count} holes detected (max confidence {max_conf:.1%})"
    else:
        message = "Fabric is clean"

    # Draw and save annotated image
    annotated_url = None
    if status:
        img     = _read_image(image_path)
        img     = _draw_boxes(img, dets)
        ann_name = "annotated_" + Path(image_path).name
        ann_path = os.path.join(ANN_DIR, ann_name)
        cv2.imwrite(ann_path, img)
        annotated_url = f"/uploads/annotated/{ann_name}"

    return {
        "hole_count":       hole_count,
        "max_confidence":   max_conf,
        "detection_status": status,
        "detections":       dets,
        "annotated_url":    annotated_url,
        "message":          message,
        "error":            False,
    }


# ── Video Detection ────────────────────────────────────────────────────────────
def detect_video(video_path: str, threshold: float = THRESHOLD,
                 frame_skip: int = 5) -> dict:
    """
    Scan a video frame-by-frame and return on the FIRST frame containing a hole.

    Returns:
        detection_status  : bool
        hole_count        : int
        max_confidence    : float
        timestamp         : str  "MM:SS"
        timestamp_sec     : float
        frame_number      : int
        frame_url         : str | None  (annotated detected frame)
        message           : str
        error             : bool
    """
    try:
        model  = _load_model()
        device = _get_device()
    except FileNotFoundError as e:
        return {"detection_status": False, "message": str(e), "error": True}
    except Exception as e:
        return {"detection_status": False, "message": f"Load error: {e}", "error": True}

    from PIL import Image as _PIL

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"detection_status": False, "message": "Cannot open video file", "error": True}

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    result    = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            # Convert BGR → PIL for Ultralytics
            pil_img = _PIL.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            dets    = _run_inference(model, pil_img, device, threshold)

            if dets:
                timestamp_sec = frame_idx / fps
                mins = int(timestamp_sec // 60)
                secs = int(timestamp_sec % 60)
                ts   = f"{mins:02d}:{secs:02d}"

                # Save annotated frame
                annotated = _draw_boxes(frame.copy(), dets)
                frame_fname = f"frame_{frame_idx}.jpg"
                cv2.imwrite(os.path.join(FRAMES_DIR, frame_fname), annotated)

                result = {
                    "detection_status": True,
                    "hole_count":       len(dets),
                    "max_confidence":   round(max(d["confidence"] for d in dets), 4),
                    "timestamp":        ts,
                    "timestamp_sec":    round(timestamp_sec, 2),
                    "frame_number":     frame_idx,
                    "frame_url":        f"/uploads/detected_frames/{frame_fname}",
                    "message":          f"Hole detected at {ts}",
                    "error":            False,
                }
                break
    finally:
        cap.release()

    if result:
        return result

    return {
        "detection_status": False,
        "hole_count":       0,
        "max_confidence":   0.0,
        "timestamp":        None,
        "frame_url":        None,
        "message":          "Fabric is clean",
        "error":            False,
    }


# ── Webcam Stream ──────────────────────────────────────────────────────────────
def generate_webcam_frames(threshold: float = THRESHOLD, camera_index: int = 0, state: dict = None):
    """
    Generator that yields MJPEG bytes for live webcam detection.
    Usage in FastAPI:
        return StreamingResponse(generate_webcam_frames(), media_type="multipart/x-mixed-replace;boundary=frame")
    """

    def _error_frame(text: str):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, text, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        _, buf = cv2.imencode(".jpg", img)
        return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

    try:
        model  = _load_model()
        device = _get_device()
    except FileNotFoundError:
        yield _error_frame("Model not trained yet")
        return

    from PIL import Image as _PIL

    # CAP_DSHOW = DirectShow backend — much faster open on Windows (skips UWP init)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    # Request MJPEG from the camera for lower USB bandwidth & faster frame delivery
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Keep only the latest frame in the buffer → no stale/delayed frames
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        yield _error_frame("No camera found")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pil_img = _PIL.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            dets    = _run_inference(model, pil_img, device, threshold)

            # Draw boxes
            frame = _draw_boxes(frame, dets)

            # Update shared state for /webcam-stats endpoint
            hole_count = len(dets)
            max_conf   = max((d["confidence"] for d in dets), default=0.0)
            if state is not None:
                state["hole_count"]    = hole_count
                state["max_confidence"] = round(max_conf, 4)

            # HUD bar
            status_text  = f"HOLES: {hole_count}  ({max_conf:.0%})" if dets else "CLEAN"
            bar_color    = (0, 0, 220) if dets else (0, 180, 0)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 40), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, f"FabricGuard  |  {status_text}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
    finally:
        cap.release()
