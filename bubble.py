from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import os
from pathlib import Path

# Directory where downloaded model will be stored
MODEL_DIR = "cached_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Cache for currently loaded model
current_model = None
current_model_name = None

def load_model(model_path=None):
    """Load the YOLO model, downloading from Hugging Face if needed"""
    global current_model, current_model_name
    
    if model_path is None:
        # Download model from Hugging Face
        print("Downloading model from Hugging Face...")
        model_path = hf_hub_download(
            repo_id="kitsumed/yolov8m_seg-speech-bubble",
            filename="model.pt",
            local_dir=MODEL_DIR
        )
        print(f"Model downloaded to: {model_path}")
    
    # Check if this is the same model already loaded
    if model_path == current_model_name:
        return current_model
    
    # Load the model
    print("Loading YOLO model...")
    current_model = YOLO(model_path)
    current_model_name = model_path
    return current_model

def predict_image(image_path, conf_threshold=0.25, iou_threshold=0.45, save_output=True, output_dir="output"):
    """
    Run speech bubble detection on an image
    
    Args:
        image_path: Path to input image
        conf_threshold: Confidence threshold (0-1)
        iou_threshold: IoU threshold for NMS (0-1)
        save_output: Whether to save the output image
        output_dir: Directory to save output images
    
    Returns:
        results: YOLO results object
        annotated_image: PIL Image with annotations
    """
    model = load_model()
    
    # Run prediction
    print(f"Running inference on: {image_path}")
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
        save=save_output,
        project=output_dir,
    )
    
    # Get annotated image from results
    for r in results:
        # r.plot() returns BGR numpy array, convert to RGB PIL Image
        im_array = r.plot()
        annotated_image = Image.fromarray(im_array[..., ::-1])  # BGR to RGB
        
        # Print detection info
        print(f"Found {len(r.boxes)} speech bubbles")
        for i, box in enumerate(r.boxes):
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            print(f"  Bubble {i+1}: confidence={conf:.3f}, class={cls}")
        
        return r, annotated_image
    
    return None, None

def get_detections(image_path, conf_threshold=0.25, iou_threshold=0.45):
    """
    Get detection results as a list of dictionaries
    
    Returns:
        List of detections with 'bbox', 'confidence', 'class', and 'mask' keys
    """
    model = load_model()
    
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
    )
    
    detections = []
    for r in results:
        for box, mask in zip(r.boxes, r.masks.data if r.masks is not None else [None] * len(r.boxes)):
            # Get bounding box coordinates (xyxy format)
            bbox = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            detection = {
                'bbox': bbox,  # [x1, y1, x2, y2]
                'confidence': conf,
                'class': cls
            }
            
            # Add mask if available
            if mask is not None:
                detection['mask'] = mask.cpu().numpy()
            
            detections.append(detection)
    
    return detections

def is_box_inside(box1, box2, threshold=0.0):
    """
    Check if box1 is inside box2 (or vice versa) with optional threshold for touching
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        threshold: Distance threshold for considering boxes as touching
    
    Returns:
        True if one box is inside the other or they're touching
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Check if box1 is inside box2 (with threshold)
    if (x1_1 >= x1_2 - threshold and y1_1 >= y1_2 - threshold and
        x2_1 <= x2_2 + threshold and y2_1 <= y2_2 + threshold):
        return True
    
    # Check if box2 is inside box1 (with threshold)
    if (x1_2 >= x1_1 - threshold and y1_2 >= y1_1 - threshold and
        x2_2 <= x2_1 + threshold and y2_2 <= y2_1 + threshold):
        return True
    
    return False

def are_boxes_touching(box1, box2, threshold=10):
    """
    Check if two boxes are touching or overlapping
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        threshold: Pixel threshold for considering boxes as touching
    
    Returns:
        True if boxes are touching or overlapping
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Check for overlap or proximity
    # Boxes overlap if: x1_1 < x2_2 + threshold and x2_1 + threshold > x1_2
    #                   and y1_1 < y2_2 + threshold and y2_1 + threshold > y1_2
    overlap_x = x1_1 < x2_2 + threshold and x2_1 + threshold > x1_2
    overlap_y = y1_1 < y2_2 + threshold and y2_1 + threshold > y1_2
    
    return overlap_x and overlap_y

def merge_boxes(box1, box2):
    """Merge two boxes into a single bounding box"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Take the union (min of mins, max of maxes)
    merged = [
        min(x1_1, x1_2),
        min(y1_1, y1_2),
        max(x2_1, x2_2),
        max(y2_1, y2_2)
    ]
    return merged

def combine_overlapping_bubbles(detections, touch_threshold=10):
    """
    Combine speech bubble bounding boxes that are inside one another or touching
    
    Args:
        detections: List of detections with 'bbox', 'confidence', etc.
        touch_threshold: Pixel threshold for considering boxes as touching
    
    Returns:
        List of merged detections
    """
    if not detections:
        return detections
    
    # Create a copy to work with
    remaining = detections.copy()
    merged = []
    
    while remaining:
        # Start with the first box
        current = remaining.pop(0)
        current_bbox = current['bbox']
        current_confidences = [current['confidence']]
        
        # Find all boxes that are inside or touching the current box
        to_merge = []
        i = 0
        while i < len(remaining):
            other = remaining[i]
            other_bbox = other['bbox']
            
            # Check if boxes are inside each other or touching
            if is_box_inside(current_bbox, other_bbox, threshold=touch_threshold) or \
               are_boxes_touching(current_bbox, other_bbox, threshold=touch_threshold):
                to_merge.append(other)
                remaining.pop(i)
                current_confidences.append(other['confidence'])
            else:
                i += 1
        
        # Merge all boxes
        if to_merge:
            # Merge bounding boxes
            for other in to_merge:
                current_bbox = merge_boxes(current_bbox, other['bbox'])
            
            # Use the maximum confidence
            max_confidence = max(current_confidences)
            
            # Create merged detection
            merged_detection = {
                'bbox': current_bbox,
                'confidence': max_confidence,
                'class': current['class'],
                'merged_count': len(to_merge) + 1  # Track how many boxes were merged
            }
            
            # Preserve mask if available (use the first one)
            if 'mask' in current:
                merged_detection['mask'] = current['mask']
            
            merged.append(merged_detection)
            print(f"  Merged {len(to_merge) + 1} overlapping/touching bubbles into one")
        else:
            # No merging needed, keep as is
            merged.append(current)
    
    return merged

def crop_and_save_bubbles(image_path, detections, output_dir="output", prefix="bubble"):
    """
    Crop speech bubbles from image using bounding boxes and save them
    
    Args:
        image_path: Path to input image
        detections: List of detections from get_detections()
        output_dir: Directory to save cropped images
        prefix: Prefix for saved filenames (e.g., "bubble" -> "bubble_0.png", "bubble_1.png")
    
    Returns:
        List of paths to saved cropped images
    """
    # Load original image
    image = Image.open(image_path).convert("RGB")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for i, det in enumerate(detections):
        bbox = det['bbox']  # [x1, y1, x2, y2]
        conf = det['confidence']
        
        # Convert to integers and ensure within image bounds
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(image.width, int(bbox[2]))
        y2 = min(image.height, int(bbox[3]))
        
        # Skip if invalid bounding box
        if x2 <= x1 or y2 <= y1:
            print(f"  Skipping invalid bbox {i+1}: {bbox}")
            continue
        
        # Crop the image
        cropped = image.crop((x1, y1, x2, y2))
        
        # Save with confidence in filename
        filename = f"{prefix}_{i+1}_conf{conf:.3f}.png"
        output_path = Path(output_dir) / filename
        cropped.save(output_path)
        saved_paths.append(str(output_path))
        
        print(f"  Saved bubble {i+1}: {output_path} (confidence: {conf:.3f})")
    
    return saved_paths

# Helper functions are available for import
# Use predict_image() or get_detections() as needed
