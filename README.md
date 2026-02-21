# FabricGuard — Hole Detection System

Single-class fabric hole detector using **YOLOv8m**.

---

## Run

```bat
# First time
setup.bat

# Every time
python main.py     →     http://127.0.0.1:8000
```

---

## Data Structure

```
data/
├── hole_image/       ← your fabric images (.jpg / .png)
└── value_image/      ← YOLO label .txt files (same filenames)
```

### Label format (each `.txt` file):
```
0 x_center y_center width height
```
- Values normalized **0.0 – 1.0**
- Class `0` = hole (the only class)
- **Empty `.txt`** = no hole in that image

---

## Modules

| File | Purpose |
|---|---|
| `main.py` | FastAPI web app — run this |
| `train.py` | YOLOv8m training logic |
| `detect.py` | Image / video / webcam detection |
| `templates/index.html` | Web UI |

---

## Training Settings

| Setting | Value |
|---|---|
| Model | yolov8m.pt (pretrained) |
| Epochs | 100 (configurable) |
| Image size | 640 × 640 |
| Batch | Auto (GPU) / 8 (CPU) |
| Optimizer | AdamW |
| Early stop | patience = 20 |
| Confidence threshold | 0.40 |

---

## Detection Output

**Hole found:**
```
"Hole detected (92.4%)"
Red bounding box drawn on image
hole_count: 2
max_confidence: 0.924
detection_status: true
```

**No hole:**
```
"Fabric is clean"
detection_status: false
```
