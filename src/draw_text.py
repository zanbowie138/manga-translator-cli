from PIL import Image, ImageDraw, ImageFont
import textwrap
import platform
import os
import cv2
import numpy as np

def _get_system_font_path():
    """Try to find a system font path"""
    system = platform.system()
    
    if system == "Windows":
        # Common Windows fonts
        font_dirs = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\Arial.ttf",
            r"C:\Windows\Fonts\verdana.ttf",
            r"C:\Windows\Fonts\Verdana.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            r"C:\Windows\Fonts\Calibri.ttf",
        ]
    elif system == "Darwin":  # macOS
        font_dirs = [
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    else:  # Linux
        font_dirs = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
    
    for font_path in font_dirs:
        if os.path.exists(font_path):
            return font_path
    
    return None

def get_font_size_for_box(text, bbox_width, bbox_height, font_path=None, max_size=100, min_size=5):
    """
    Find the largest font size that fits text within a bounding box
    
    Args:
        text: Text to fit
        bbox_width: Width of bounding box
        bbox_height: Height of bounding box
        font_path: Path to font file (None to try system font)
        max_size: Maximum font size to try
        min_size: Minimum font size to try
    
    Returns:
        PIL ImageFont object at appropriate size
    """
    # Try to find a system font if no path provided
    if font_path is None:
        font_path = _get_system_font_path()
    
    # Create a temporary image and draw object for measuring actual text bounds
    temp_image = Image.new('RGB', (bbox_width, bbox_height), color='white')
    temp_draw = ImageDraw.Draw(temp_image)
    
    def text_fits(size):
        """Check if text fits within bounds at the given font size"""
        try:
            font = ImageFont.truetype(font_path, size)
            
            # Calculate wrapped text dimensions
            avg_char_width = font.getlength('M')
            chars_per_line = max(1, int(bbox_width * 0.80 / avg_char_width))
            wrapped_lines = textwrap.wrap(text, width=chars_per_line, break_long_words=False)
            
            # Measure actual bounding box of rendered text using ImageDraw
            line_height = font.getbbox('Ay')[3]  # Get line height for spacing
            max_line_width = 0
            total_text_height = 0
            current_y = 0
            
            # Calculate actual width and height by measuring each line's bbox
            for line in wrapped_lines:
                # Get actual bounding box of the rendered line
                line_bbox = temp_draw.textbbox((0, current_y), line, font=font)
                line_width = line_bbox[2] - line_bbox[0]
                line_height_actual = line_bbox[3] - line_bbox[1]
                max_line_width = max(max_line_width, line_width)
                total_text_height = current_y + line_height_actual
                current_y += line_height * 1.15  # Add spacing between lines
            
            # Check if actual rendered text fits within bounds
            if max_line_width <= bbox_width * 0.80 and total_text_height <= bbox_height * 0.80:
                return font, wrapped_lines
        except Exception:
            pass
        return None, None
    
    # Binary search for optimal font size
    left = min_size
    right = max_size
    best_font = None
    best_wrapped_lines = None
    
    while left <= right:
        mid = (left + right) // 2
        font, wrapped_lines = text_fits(mid)
        
        if font is not None:
            # This size fits, try larger
            best_font = font
            best_wrapped_lines = wrapped_lines
            left = mid + 1
        else:
            # This size doesn't fit, try smaller
            right = mid - 1
    
    # If we found a font, use it
    if best_font and best_wrapped_lines:
        return best_font, best_wrapped_lines
    
    # Fallback: calculate based on dimensions
    fallback_size = int(min(bbox_width, bbox_height) * 0.20)
    fallback_size = max(min_size, min(fallback_size, max_size))
    
    font = ImageFont.truetype(font_path, fallback_size)
    avg_char_width = font.getlength('M')
    
    chars_per_line = max(1, int(bbox_width * 0.90 / avg_char_width))
    wrapped_lines = textwrap.wrap(text, width=chars_per_line)

    print(f"Failed to find a good font size, using fallback size: {fallback_size}")
    
    return font, wrapped_lines

def draw_text_with_outline(draw, text, position, font, fill_color='black', outline_color='white', outline_width=10):
    """
    Draw text with an outline (stroke)
    
    Args:
        draw: PIL ImageDraw object
        text: Text to draw
        position: (x, y) tuple for text position
        font: PIL ImageFont object
        fill_color: Text fill color
        outline_color: Outline/stroke color
        outline_width: Width of outline in pixels
    """
    x, y = position
    
    # Draw outline by drawing text multiple times with offsets
    for adj in range(-outline_width, outline_width + 1):
        for adj2 in range(-outline_width, outline_width + 1):
            if adj == 0 and adj2 == 0:
                continue
            draw.text((x + adj, y + adj2), text, font=font, fill=outline_color)
    
    # Draw main text on top
    draw.text(position, text, font=font, fill=fill_color)

def draw_text_on_image(image_path, box_texts, bubble_masks, font_path=None):
    """
    Draw translated text on the original image within bubble interior shapes
    
    Args:
        image_path: Path to original image
        box_texts: List of (bbox, translated_text) tuples where bbox is [x1, y1, x2, y2]
        bubble_masks: Dictionary mapping bbox tuple to bubble_mask numpy array
        font_path: Path to font file (None to try system font)
    
    Returns:
        PIL Image with translated text drawn on it
    """
    # Load image as PIL for drawing
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    for bbox, translated_text in box_texts:
        if not translated_text or not translated_text.strip():
            continue
        
        # bbox is [x1, y1, x2, y2]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Get the mask for this bubble
        bbox_key = tuple(bbox)
        if bbox_key not in bubble_masks:
            continue
        
        bubble_mask = bubble_masks[bbox_key]
        
        # Find bounding rectangle of the mask (actual interior shape)
        mask_points = cv2.findNonZero(bubble_mask)
        if mask_points is None or len(mask_points) == 0:
            # Fallback to bounding box if no mask found
            mask_x, mask_y, mask_w, mask_h = 0, 0, x2 - x1, y2 - y1
        else:
            mask_rect = cv2.boundingRect(mask_points)
            mask_x, mask_y, mask_w, mask_h = mask_rect
        
        # Calculate actual interior dimensions
        interior_width = mask_w
        interior_height = mask_h
        
        # Skip if interior is too small
        if interior_width < 20 or interior_height < 20:
            continue
        
        # Get font and wrapped text that fits in the interior
        font, wrapped_lines = get_font_size_for_box(translated_text, interior_width, interior_height, font_path=font_path)
        
        # Calculate line height
        line_height = font.getbbox('Ay')[3] * 1.15
        total_text_height = len(wrapped_lines) * line_height
        
        # Calculate text position aligned to interior shape
        # Position relative to bubble crop, then offset by bubble position in image
        interior_start_x = x1 + mask_x
        interior_start_y = y1 + mask_y
        interior_center_x = interior_start_x + interior_width / 2
        interior_center_y = interior_start_y + interior_height / 2
        
        # Center text vertically within interior
        start_y = interior_center_y - total_text_height / 2
        
        # Draw each line of wrapped text
        current_y = start_y
        for line in wrapped_lines:
            # Center text horizontally within the interior
            line_width = font.getlength(line)
            start_x = interior_center_x - line_width / 2
            
            # Draw text with white outline
            draw_text_with_outline(
                draw,
                line,
                (int(start_x), int(current_y)),
                font,
                fill_color='black',
                outline_color='white',
                outline_width=2
            )
            
            current_y += line_height
    
    return image

