"""
YOLOv8n Face Detection Validation Script
This script validates the trained model on the test/validation dataset
"""

from ultralytics import YOLO
import os

# Load the best trained model
print("Loading best trained model...")
model_path = 'runs/detect/face_detection_v1/weights/best.pt'

# Check if model exists
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    print("Please train the model first using: python train_yolov8n.py")
    exit(1)

model = YOLO(model_path)

# Validate the model
print("Validating model on test dataset...")
metrics = model.val(
    data='data.yaml',
    split='test',          # Use test split
    imgsz=640,
    batch=16,
    save_json=True,        # Save results in COCO JSON format
    save_hybrid=True,      # Save hybrid version of labels
    conf=0.25,             # Confidence threshold
    iou=0.6,               # IoU threshold for NMS
    max_det=300,           # Maximum detections per image
    plots=True,            # Save validation plots
    device=0,              # Use GPU 0 (change to 'cpu' if no GPU)
)

# Print validation metrics
print("\n" + "="*50)
print("Validation Results:")
print("="*50)
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print("="*50)
