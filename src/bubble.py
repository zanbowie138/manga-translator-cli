from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import os

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

def find_speech_bubbles(image_path, conf_threshold=0.25, iou_threshold=0.45, save_output=True, output_dir="output"):
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

def get_cropped_images(image_path, detections):
    """
    Get cropped PIL Images for each detection bounding box
    
    Args:
        image_path: Path to input image
        detections: List of detections with 'bbox' keys
    
    Returns:
        List of (cropped_image, detection) tuples
    """
    # Load original image
    image = Image.open(image_path).convert("RGB")
    
    cropped_images = []
    
    for i, det in enumerate(detections):
        bbox = det['bbox']  # [x1, y1, x2, y2]
        
        # Convert to integers and ensure within image bounds
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(image.width, int(bbox[2]))
        y2 = min(image.height, int(bbox[3]))
        
        # Skip if invalid bounding box
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Crop the image
        cropped = image.crop((x1, y1, x2, y2))
        cropped_images.append((cropped, det))
    
    return cropped_images

