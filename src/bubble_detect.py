from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import os
import numpy as np
from src.bbox import BoundingBox

# Directory where downloaded model will be stored
MODEL_DIR = "cached_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Cache for currently loaded model
current_model = None
current_model_name = None

def load_bubble_detection_model(model_path=None):
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

def run_detection(model, image: Image.Image, conf_threshold=0.25, iou_threshold=0.45, silent=False):
    """
    Run speech bubble detection on an image and process the results.
    
    Args:
        model: Loaded YOLO model
        image: Input PIL Image
        conf_threshold: Confidence threshold (0-1)
        iou_threshold: IoU threshold for NMS (0-1)
        silent: If True, suppress progress messages
    
    Returns:
        Tuple of (annotated_bubble_image, boxes) where:
        - annotated_bubble_image: PIL Image with annotations (or None if no detections)
        - boxes: List of BoundingBox instances (or empty list if no detections)
    """
    # Convert PIL Image to numpy array for YOLO
    img_array = np.array(image.convert("RGB"))
    
    # Run prediction (without saving to avoid creating predict/ directories)
    if not silent:
        print("Running inference on image...")
    results = model.predict(
        source=img_array,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
        save=False,
    )
    
    # Process results to extract annotated image and bounding boxes
    for r in results:
        # r.plot() returns BGR numpy array, convert to RGB PIL Image
        im_array = r.plot()
        annotated_bubble_image = Image.fromarray(im_array[..., ::-1])  # BGR to RGB
        
        # Extract bounding boxes
        boxes = []
        for box in r.boxes:
            bbox_list = box.xyxy[0].cpu().numpy().tolist()
            boxes.append(BoundingBox.from_list(bbox_list))
        
        # Print detection info
        if not silent:
            print(f"Found {len(boxes)} speech bubbles")
        for i, box in enumerate(r.boxes):
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if not silent:
                print(f"  Bubble {i+1}: confidence={conf:.3f}, class={cls}")
        
        return annotated_bubble_image, boxes
    
    return None, []

def get_detections(image_path, conf_threshold=0.25, iou_threshold=0.45):
    """
    Get detection results as a list of bounding boxes
    
    Returns:
        List of BoundingBox instances
    """
    model = load_bubble_detection_model()
    
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
    )
    
    boxes = []
    for r in results:
        for box in r.boxes:
            # Get bounding box coordinates (xyxy format)
            bbox_list = box.xyxy[0].cpu().numpy().tolist()
            boxes.append(BoundingBox.from_list(bbox_list))
    
    return boxes


