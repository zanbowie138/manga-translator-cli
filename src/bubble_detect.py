from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import os
import numpy as np
from src.bbox import BoundingBox

# Directory where downloaded model will be stored
MODEL_DIR = os.path.join(os.path.expanduser("~"), ".manga-translate", "cached_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Cache for currently loaded model
_cached_model = None
_cached_key = None

def _normalize_model_path(model_path):
    """Normalize model path for cache comparison."""
    if model_path is None:
        return None
    expanded = os.path.expanduser(model_path)
    return os.path.normpath(os.path.abspath(expanded))

def load_bubble_detection_model(model_path=None, silent=False):
    """Load the YOLO model, downloading from Hugging Face if needed"""
    global _cached_model, _cached_key

    if model_path is None:
        # Download model from Hugging Face (hf_hub_download shows its own progress)
        if not silent:
            print(f"Downloading YOLO model to: {MODEL_DIR}")
        model_path = hf_hub_download(
            repo_id="kitsumed/yolov8m_seg-speech-bubble",
            filename="model.pt",
            local_dir=MODEL_DIR
        )
        if not silent:
            print(f"Model downloaded to: {model_path}")

    # Normalize path for cache comparison
    normalized_path = _normalize_model_path(model_path)

    # Check if this is the same model already loaded
    if _cached_model is not None and normalized_path == _cached_key:
        if not silent:
            print(f"Using cached YOLO model from: {model_path}")
        return _cached_model

    # Load the model
    if not silent:
        print(f"Loading YOLO model from: {model_path}")
    _cached_model = YOLO(model_path)
    _cached_key = normalized_path
    return _cached_model

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
    results = model.predict(
        source=img_array,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
        save=False,
        verbose=not silent,
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


