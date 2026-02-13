"""
YOLOv8n Face Detection Prediction Script
This script runs inference on new images using the trained model
"""

from ultralytics import YOLO
import argparse
import os
import cv2
from pathlib import Path

def crop_and_save_faces(result, source_info, output_dir, is_video=False):
    """
    Crop detected faces and save them as separate images.
    
    Args:
        result: YOLO result object containing detection boxes
        source_info: str - image name (for images) or int - frame number (for videos)
        output_dir: str - output directory path
        is_video: bool - whether source is video or image
    """
    # Get the original image from the result
    orig_img = result.orig_img
    
    # Get image dimensions for coordinate clipping
    img_height, img_width = orig_img.shape[:2]
    
    # Check if any faces were detected
    if len(result.boxes) == 0:
        return 0
    
    # Get bounding boxes in xyxy format (x1, y1, x2, y2)
    boxes = result.boxes.xyxy.cpu().numpy()
    
    saved_count = 0
    
    # Iterate through each detected face
    for idx, box in enumerate(boxes, start=1):
        # Extract coordinates and clip to image boundaries
        x1, y1, x2, y2 = box
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(img_width, x2))
        y2 = int(min(img_height, y2))
        
        # Skip if invalid box (no area)
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Crop the face region
        face_crop = orig_img[y1:y2, x1:x2]
        
        # Convert from BGR to RGB (OpenCV uses BGR by default)
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Generate filename based on source type
        if is_video:
            # For video: frameNumber_face_N.jpg
            filename = f"frame_{source_info:04d}_face_{idx}.jpg"
        else:
            # For image: imageName_face_N.jpg
            base_name = Path(source_info).stem  # Get filename without extension
            filename = f"{base_name}_face_{idx}.jpg"
        
        # Full output path
        output_path = os.path.join(output_dir, filename)
        
        # Save the cropped face as JPG (convert RGB back to BGR for cv2.imwrite)
        cv2.imwrite(output_path, cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR))
        saved_count += 1
    
    return saved_count

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
    parser.add_argument('--save-faces', action='store_true', default=False,
                        help='Enable face cropping and saving feature')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train the model first using: python train_yolov8n.py")
        return
    
    # Create output directory for cropped faces if --save-faces is enabled
    faces_output_dir = "detected_faces"
    if args.save_faces:
        os.makedirs(faces_output_dir, exist_ok=True)
        print(f"Face crops will be saved to: {faces_output_dir}/")
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    # Determine if source is a video
    is_video = args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
    
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
    
    # Process face cropping if enabled
    total_faces_saved = 0
    if args.save_faces:
        print("\nCropping and saving detected faces...")
        for i, result in enumerate(results):
            if is_video:
                # For video, use frame number
                source_info = i
            else:
                # For image, use the source path
                source_info = args.source
            
            faces_saved = crop_and_save_faces(result, source_info, faces_output_dir, is_video)
            total_faces_saved += faces_saved
    
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
    
    if args.save_faces:
        print(f"\nTotal face crops saved: {total_faces_saved}")
        print(f"Face crops location: {faces_output_dir}/")
    
    print("="*50)

if __name__ == '__main__':
    main()
