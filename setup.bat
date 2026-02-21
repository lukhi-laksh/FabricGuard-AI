@echo off
echo =====================================================
echo  FabricGuard  -  Hole Detection System  Setup
echo =====================================================
echo.

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

echo.
echo Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo Installing GPU-enabled PyTorch (CUDA 12.1)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

echo.
echo Downloading YOLOv8m weights...
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt'); print('yolov8m.pt ready')"

echo.
echo =====================================================
echo  Setup complete!
echo  Run:  python main.py
echo  UI:   http://127.0.0.1:8000
echo =====================================================
pause