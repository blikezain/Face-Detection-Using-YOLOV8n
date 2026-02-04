"""
YOLOv8n Face Detection Training Script
This script trains a YOLOv8 nano model on the face detection dataset
"""

from ultralytics import YOLO
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the YOLOv8n model
print("Loading YOLOv8n model...")
model = YOLO('yolov8n.pt')

# Train the model
print("Starting training...")
results = model.train(
    data='data.yaml',           # Path to data configuration file
    epochs=50,                   # Number of training epochs
    imgsz=640,                   # Image size (already preprocessed to 640x640)
    batch=16,                    # Batch size (adjust based on your GPU memory)
    name='face_detection_v1',    # Name of the training run
    patience=10,                 # Early stopping patience
    save=True,                   # Save checkpoints
    device=0,                    # Use GPU 0 (change to 'cpu' if no GPU)
    workers=4,                   # Number of worker threads
    project='runs/detect',       # Project directory
    exist_ok=True,               # Allow overwriting existing runs
    pretrained=True,             # Use pretrained weights
    optimizer='auto',            # Optimizer (auto, SGD, Adam, AdamW)
    verbose=True,                # Verbose output
    seed=42,                     # Random seed for reproducibility
    deterministic=True,          # Make training deterministic
    single_cls=True,             # Single class (Face only)
    rect=False,                  # Rectangular training
    cos_lr=False,                # Cosine learning rate scheduler
    close_mosaic=10,             # Disable mosaic augmentation for last N epochs
    resume=False,                # Resume training from last checkpoint
    amp=True,                    # Automatic Mixed Precision training
    fraction=1.0,                # Fraction of dataset to use
    profile=False,               # Profile ONNX and TensorRT speeds
    freeze=None,                 # Freeze layers (None or list of layer indices)
    lr0=0.01,                    # Initial learning rate
    lrf=0.01,                    # Final learning rate (lr0 * lrf)
    momentum=0.937,              # SGD momentum/Adam beta1
    weight_decay=0.0005,         # Optimizer weight decay
    warmup_epochs=3.0,           # Warmup epochs
    warmup_momentum=0.8,         # Warmup initial momentum
    warmup_bias_lr=0.1,          # Warmup initial bias lr
    box=7.5,                     # Box loss gain
    cls=0.5,                     # Classification loss gain
    dfl=1.5,                     # Distribution focal loss gain
    plots=True,                  # Save plots during training
    save_period=-1,              # Save checkpoint every x epochs (-1 = only last)
)

# Print training results
print("\n" + "="*50)
print("Training completed!")
print("="*50)
print(f"Best model saved at: {results.save_dir}/weights/best.pt")
print(f"Last model saved at: {results.save_dir}/weights/last.pt")
print(f"Results saved in: {results.save_dir}")
print("\nTo validate the model, run:")
print(f"python validate_model.py")
print("\nTo make predictions, run:")
print(f"python predict.py --source <image_path>")
