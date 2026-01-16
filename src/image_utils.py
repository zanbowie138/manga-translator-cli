from pathlib import Path
from PIL import Image
from src.bbox import BoundingBox, normalize_boxes


def save_pil_image(image: Image.Image, output_path: str, print_message: bool = True) -> None:
    """
    Save a PIL Image to the specified path, creating parent directories if needed.
    
    Args:
        image: PIL Image to save
        output_path: Path where the image should be saved
        print_message: If True, print a confirmation message after saving
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    
    if print_message:
        print(f"Image saved to: {output_path}")


def get_cropped_images(image_path, boxes):
    """
    Get cropped PIL Images for each bounding box
    
    Args:
        image_path: Path to input image
        boxes: List of bounding boxes (can be lists or BoundingBox instances)
    
    Returns:
        List of (cropped_image, bbox) tuples where bbox is a BoundingBox instance
    """
    # Load original image
    image = Image.open(image_path).convert("RGB")
    
    # Normalize to BoundingBox instances
    normalized_boxes = normalize_boxes(boxes)
    
    cropped_images = []
    
    for bbox in normalized_boxes:
        # Clip to image bounds
        clipped_bbox = bbox.clip(image.width, image.height)
        
        # Skip if invalid bounding box
        if not clipped_bbox.is_valid():
            continue
        
        # Crop the image
        x1, y1, x2, y2 = clipped_bbox
        cropped = image.crop((x1, y1, x2, y2))
        cropped_images.append((cropped, bbox))  # Keep original bbox, not clipped
    
    return cropped_images