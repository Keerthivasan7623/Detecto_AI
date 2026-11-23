# PPE Model Setup

The app expects a YOLO checkpoint at `models/best.pt`. No model weights are included in this ZIP.

## Recommended route
1. Read the official Ultralytics Construction-PPE dataset documentation:
   https://docs.ultralytics.com/datasets/detect/construction-ppe/
2. Download the dataset using the current official instructions, including its YAML configuration.
3. Train a YOLOv8 model using the dataset YAML's actual local path.
4. Copy the resulting `best.pt` to `models/best.pt`.

Example command after you have the local dataset YAML:
```powershell
yolo detect train model=yolov8n.pt data="C:\path\to\construction-ppe.yaml" epochs=30 imgsz=640 batch=8 device=0
```

Adjust batch size to fit GPU memory. If CUDA out-of-memory occurs, reduce `batch` to 4 or 2.

Verify GPU availability:
```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Important: use the model's actual class names. This project only flags missing-PPE labels that are explicitly listed in `app.py`; it does not infer that PPE is missing merely because a helmet or vest detection is absent.
