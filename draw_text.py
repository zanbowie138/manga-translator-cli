from PIL import Image, ImageDraw, ImageFont
import textwrap
from pathlib import Path
import platform

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
        if Path(font_path).exists():
            return font_path
    
    return None

def get_font_size_for_box(text, bbox_width, bbox_height, font_path=None, max_size=100, min_size=24):
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
    
    # Calculate a good starting size based on bounding box
    # Estimate: use about 20-25% of the smaller dimension for better proportion
    base_size = int(min(bbox_width, bbox_height) * 0.22)
    start_size = min(max_size, max(base_size, min_size))
    
    # Try to load a font with size
    best_font = None
    best_wrapped_lines = None
    best_size = min_size
    
    for size in range(start_size, min_size - 1, -5):
        try:
            if font_path:
                font = ImageFont.truetype(font_path, size)
            else:
                # Use default font - but note it doesn't scale well
                font = ImageFont.load_default()
            
            # Calculate wrapped text dimensions
            # Estimate characters per line: assume average char width is about 0.6 * font size
            if font_path:
                avg_char_width = font.getlength('M')
            else:
                # For default font, estimate based on size
                avg_char_width = size * 0.6
            
            chars_per_line = max(1, int(bbox_width * 0.90 / avg_char_width))
            wrapped_lines = textwrap.wrap(text, width=chars_per_line)
            
            # Calculate total height needed
            if font_path:
                line_height = font.getbbox('Ay')[3]  # Get line height
            else:
                line_height = size * 1.2  # Estimate for default font
            
            total_height = len(wrapped_lines) * line_height * 1.15  # 1.15 for spacing
            
            # Check if it fits
            if total_height <= bbox_height * 0.90:
                best_font = font
                best_wrapped_lines = wrapped_lines
                best_size = size
                break  # Found a good size, use it
        
        except Exception as e:
            continue
    
    # If we found a font, use it
    if best_font and best_wrapped_lines:
        return best_font, best_wrapped_lines
    
    # Fallback: calculate based on dimensions
    fallback_size = int(min(bbox_width, bbox_height) * 0.20)
    fallback_size = max(min_size, min(fallback_size, max_size))
    
    if font_path:
        font = ImageFont.truetype(font_path, fallback_size)
        avg_char_width = font.getlength('M')
    else:
        font = ImageFont.load_default()
        avg_char_width = fallback_size * 0.6  # Estimate for default font
    
    chars_per_line = max(1, int(bbox_width * 0.90 / avg_char_width))
    wrapped_lines = textwrap.wrap(text, width=chars_per_line)
    
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

def draw_translated_text_on_image(image_path, detections_translations, output_path):
    """
    Draw translated text on the original image within speech bubble bounding boxes
    
    Args:
        image_path: Path to original image
        detections_translations: List of (detection, translated_text) tuples
        output_path: Path to save the output image
    
    Returns:
        PIL Image with translated text drawn on it
    """
    # Load original image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    for det, translated_text in detections_translations:
        if not translated_text or not translated_text.strip():
            continue
        
        bbox = det['bbox']  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Calculate bounding box dimensions
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Skip if box is too small
        if bbox_width < 20 or bbox_height < 20:
            continue
        
        # Get font and wrapped text that fits in the box
        font, wrapped_lines = get_font_size_for_box(translated_text, bbox_width, bbox_height)
        
        # Calculate line height - use less spacing
        line_height = font.getbbox('Ay')[3] * 1.15
        start_y = y1 + (bbox_height - len(wrapped_lines) * line_height) / 2
        
        # Draw each line of wrapped text
        current_y = start_y
        for line in wrapped_lines:
            # Center text horizontally within the bounding box
            line_width = font.getlength(line)
            start_x = x1 + (bbox_width - line_width) / 2
            
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
    
    # Save the image
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    
    return image

