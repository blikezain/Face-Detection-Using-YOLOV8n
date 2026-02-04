# YOLOv8n Face Detection Training Project

This project trains a YOLOv8 nano model for face detection on a custom dataset.

## Dataset Information

- **Total Images**: 3,647 images
- **Format**: COCO format
- **Image Size**: 640×640 pixels
- **Classes**: 1 (Face)
- **Splits**: Train / Validation / Test

## Files

- `data.yaml` - Dataset configuration file
- `train_yolov8n.py` - Training script
- `validate_model.py` - Model validation script
- `predict.py` - Inference script for predictions
- `yolov8n.pt` - Pre-trained YOLOv8 nano model

## Quick Start

### 1. Train the Model

```bash
python train_yolov8n.py
```

This will:

- Train for 50 epochs
- Use batch size of 16
- Save checkpoints to `runs/detect/face_detection_v1/`
- Generate training plots and metrics

### 2. Validate the Model

```bash
python validate_model.py
```

This will validate the trained model and show:

- mAP50 and mAP50-95 scores
- Precision and Recall

### 3. Run Predictions

```bash
# On a single image
python predict.py --source face.jpg

# On a directory
python predict.py --source path/to/images/

# With custom confidence threshold
python predict.py --source face.jpg --conf 0.5
```

## Training Parameters

- **Epochs**: 50 (with early stopping patience of 10)
- **Batch Size**: 16
- **Image Size**: 640×640
- **Optimizer**: Auto (AdamW)
- **Learning Rate**: 0.01 (initial) → 0.0001 (final)
- **Device**: GPU (cuda:0) or CPU

## Results Location

All training results are saved in:

- `runs/detect/face_detection_v1/weights/best.pt` - Best model
- `runs/detect/face_detection_v1/weights/last.pt` - Last checkpoint
- `runs/detect/face_detection_v1/` - Training plots and metrics

## Requirements

```bash
pip install ultralytics
```

## Notes

- Adjust `batch` size in `train_yolov8n.py` based on your GPU memory
- Change `device=0` to `device='cpu'` if you don't have a GPU
- Training time depends on your hardware (GPU recommended)
