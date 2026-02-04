"""
YOLOv8n Face Detection Prediction Script
This script runs inference on new images using the trained model
"""

from ultralytics import YOLO
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Run face detection on images')
    parser.add_argument('--source', type=str, required=True, 
                        help='Path to image, video, or directory')
    parser.add_argument('--model', type=str, 
                        default='runs/detect/face_detection_v1/weights/best.pt',
                        help='Path to trained model')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save detection results')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train the model first using: python train_yolov8n.py")
        return
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    # Run prediction
    print(f"Running detection on {args.source}...")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=args.save,
        save_txt=True,         # Save results to txt
        save_conf=True,        # Save confidences in txt
        imgsz=640,
        device=0,              # Use GPU 0 (change to 'cpu' if no GPU)
        show=False,            # Don't display results
        project='runs/predict',
        name='face_detection',
        exist_ok=True,
        show_labels=True,
        show_conf=True,
        line_width=2,
    )
    
    # Print results
    print("\n" + "="*50)
    print("Detection completed!")
    print("="*50)
    for i, result in enumerate(results):
        print(f"\nImage {i+1}:")
        print(f"  Detections: {len(result.boxes)}")
        if len(result.boxes) > 0:
            print(f"  Confidences: {[f'{conf:.2f}' for conf in result.boxes.conf.tolist()]}")
    print(f"\nResults saved in: runs/predict/face_detection")
    print("="*50)

if __name__ == '__main__':
    main()
