from pathlib import Path
from PIL import Image


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
        boxes: List of bounding boxes as [x1, y1, x2, y2]
    
    Returns:
        List of (cropped_image, bbox) tuples
    """
    # Load original image
    image = Image.open(image_path).convert("RGB")
    
    cropped_images = []
    
    for bbox in boxes:
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
        cropped_images.append((cropped, bbox))
    
    return cropped_images