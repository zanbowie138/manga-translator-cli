import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Union


def get_bubble_text_mask(
    bubble_img: np.ndarray,
    threshold_value: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use whitespace detection to find the speech bubble outline and interior.
    Threshold: White = Paper, Black = Ink.
    The largest white blob is the speech bubble.
    
    Args:
        bubble_img: Cropped bubble image as numpy array (BGR)
        threshold_value: Threshold value for binary thresholding (default: 200)
    
    Returns:
        Tuple of (bubble_mask, text_mask) where:
        - bubble_mask: Mask of the bubble interior (255 = inside bubble, 0 = outside)
        - text_mask: Mask of text inside the bubble (255 = text, 0 = other)
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(bubble_img, cv2.COLOR_BGR2GRAY)
    
    # 2. Thresholding: White = 255 (Paper), Black = 0 (Ink)
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # 3. Find Contours
    # RETR_EXTERNAL = retrieves only the extreme outer contours
    # CHAIN_APPROX_SIMPLE = compresses horizontal, vertical, and diagonal segments
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # No white regions found, return empty masks
        bubble_mask = np.zeros_like(gray)
        text_mask = np.zeros_like(gray)
        return bubble_mask, text_mask
    
    # 4. Identify the Speech Bubble
    # Assumption: The speech bubble is the largest white object in the crop
    bubble_contour = max(contours, key=cv2.contourArea)
    
    # 5. Create a "Bubble Mask"
    # This mask represents the purely white shape of the bubble
    bubble_mask = np.zeros_like(gray)
    cv2.drawContours(bubble_mask, [bubble_contour], -1, 255, thickness=cv2.FILLED)
    
    # 6. Isolate the Text
    # Logic: "Show me pixels that are INSIDE the bubble shape (bubble_mask)
    #         BUT are black in the original image (binary)"
    
    # Invert binary to make text white (255) and background black (0)
    text_is_white = cv2.bitwise_not(binary)
    
    # Intersection: (Inside Bubble) AND (Is Text)
    text_mask = cv2.bitwise_and(bubble_mask, text_is_white)
    
    # 7. Dilate (Optional but recommended)
    # Expands the text mask slightly to catch anti-aliasing artifacts
    kernel = np.ones((3, 3), np.uint8)
    text_mask = cv2.dilate(text_mask, kernel, iterations=2)
    
    return bubble_mask, text_mask


def get_bubble_base_color_from_mask(
    bubble_crop: np.ndarray,
    bubble_mask: np.ndarray,
    exclude_text_mask: np.ndarray = None
) -> Tuple[int, int, int]:
    """
    Sample the base color of a speech bubble by sampling pixels inside the bubble mask.
    
    Args:
        bubble_crop: Cropped bubble image as numpy array (BGR)
        bubble_mask: Mask of bubble interior (255 = inside bubble, 0 = outside)
        exclude_text_mask: Optional mask to exclude text regions from sampling
    
    Returns:
        Base color as (B, G, R) tuple
    """
    # Create sampling mask: inside bubble but not text
    sampling_mask = bubble_mask.copy()
    if exclude_text_mask is not None:
        sampling_mask = cv2.bitwise_and(sampling_mask, 255 - exclude_text_mask)
    
    # Sample pixels that are inside the bubble (and not text)
    samples = bubble_crop[sampling_mask > 0]
    
    if len(samples) > 0:
        # Use median to get representative color
        median_color = np.median(samples, axis=0).astype(int)
        # Convert numpy scalars to Python ints for OpenCV
        base_color = (int(median_color[0]), int(median_color[1]), int(median_color[2]))
        return base_color
    
    # Fallback: sample from center of bubble crop
    center_y, center_x = bubble_crop.shape[0] // 2, bubble_crop.shape[1] // 2
    center_color = bubble_crop[center_y, center_x].astype(int)
    return (int(center_color[0]), int(center_color[1]), int(center_color[2]))


def color_bubble_interiors_blue(
    image: Union[str, np.ndarray, Image.Image],
    boxes: List[Tuple[float, float, float, float]],
    threshold_value: int = 200
) -> np.ndarray:
    """
    Color all speech bubble interiors blue for visualization.
    
    Args:
        image: Input image as file path, numpy array (BGR), or PIL Image
        boxes: List of bounding boxes as (x1, y1, x2, y2)
        threshold_value: Threshold value for binary thresholding (default: 200)
    
    Returns:
        Image with bubble interiors colored blue as numpy array (BGR)
    """
    # Load image
    if isinstance(image, str):
        img_array = cv2.imread(image)
        if img_array is None:
            raise ValueError(f"Could not load image from {image}")
    elif isinstance(image, Image.Image):
        img_array = np.array(image.convert("RGB"))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        img_array = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    output = img_array.copy()
    blue_color = (255, 0, 0)  # BGR format: blue
    
    # Process each bubble
    for bubble_box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in bubble_box]
        
        # Ensure bubble box is within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_array.shape[1], x2)
        y2 = min(img_array.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Crop the bubble
        bubble_crop = output[y1:y2, x1:x2].copy()
        
        # Get bubble outline and interior using whitespace detection
        bubble_mask, _ = get_bubble_text_mask(bubble_crop, threshold_value)
        
        # Color bubble interior blue
        bubble_crop[bubble_mask > 0] = blue_color
        
        # Paste the modified crop back into the output image
        output[y1:y2, x1:x2] = bubble_crop
    
    return output


def fill_bubble_interiors(
    image: Union[str, np.ndarray, Image.Image],
    boxes: List[Tuple[float, float, float, float]],
    threshold_value: int = 200,
    use_inpaint: bool = False
) -> np.ndarray:
    """
    Use whitespace detection to find speech bubble outlines and fill everything inside with the base color.
    
    Args:
        image: Input image as file path, numpy array (BGR), or PIL Image
        boxes: List of bounding boxes as (x1, y1, x2, y2)
        threshold_value: Threshold value for binary thresholding (default: 200)
        use_inpaint: If True, use inpainting for text removal. If False, fill entire interior with base color.
    
    Returns:
        Image with bubble interiors filled with base color as numpy array (BGR)
    """
    # Load image
    if isinstance(image, str):
        img_array = cv2.imread(image)
        if img_array is None:
            raise ValueError(f"Could not load image from {image}")
    elif isinstance(image, Image.Image):
        img_array = np.array(image.convert("RGB"))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        img_array = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    output = img_array.copy()
    
    # Process each bubble
    for bubble_box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in bubble_box]
        
        # Ensure bubble box is within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_array.shape[1], x2)
        y2 = min(img_array.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Crop the bubble
        bubble_crop = output[y1:y2, x1:x2].copy()
        
        # Get bubble outline and interior using whitespace detection
        bubble_mask, text_mask = get_bubble_text_mask(bubble_crop, threshold_value)
        
        # Get base color of the bubble by sampling from inside the bubble mask
        base_color = get_bubble_base_color_from_mask(bubble_crop, bubble_mask, exclude_text_mask=text_mask)
        
        if use_inpaint:
            # Option 1: Inpaint only the text regions
            bubble_crop = cv2.inpaint(bubble_crop, text_mask, 3, cv2.INPAINT_TELEA)
        else:
            # Option 2: Fill entire interior with base color
            bubble_crop[bubble_mask > 0] = base_color
        
        # Paste the modified crop back into the output image
        output[y1:y2, x1:x2] = bubble_crop
    
    return output


