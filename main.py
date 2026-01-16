from src.manga_ocr import extract_text
from src.bubble import load_model, run_detection, process_detection_results
from src.image_utils import get_cropped_images, save_pil_image
from src.bbox import remove_parent_boxes
from src.translate import translate_phrase
from src.draw_text import draw_text_on_image
from src.bubble_clean import fill_bubble_interiors, color_bubble_interiors_blue, get_bubble_text_mask
import sys
import cv2
from PIL import Image

# Image path constants
INPUT_IMAGE_PATH = "input/lupin1.png"
OUTPUT_DIR = "output"
SPEECH_BUBBLES_OUTPUT_PATH = "output/speech_bubbles.png"
BUBBLE_INTERIORS_BLUE_PATH = "output/bubble_interiors_blue.png"
CLEANED_IMAGE_PATH = "output/cleaned.png"
TRANSLATED_OUTPUT_PATH = "output/translated.png"

# Font constant
FONT_PATH = "fonts/CC Astro City Int Regular.ttf"

def main():
    """Main function that orchestrates both PaddleOCR and bubble detection"""
    
    print("\n" + "=" * 50)
    print("Running speech bubble detection...")
    print("=" * 50)
    
    # Load model once
    model = load_model()
    
    # Detect speech bubbles
    try:
        # Run detection
        results = run_detection(model, INPUT_IMAGE_PATH, conf_threshold=0.25, iou_threshold=0.45)
        
        # Process results
        annotated_img, boxes = process_detection_results(results)
        
        if annotated_img:
            # Save to custom location
            save_pil_image(annotated_img, SPEECH_BUBBLES_OUTPUT_PATH)
            
            print(f"\nTotal detections: {len(boxes)} speech bubbles found")
        
        # Remove parent boxes, keeping only children
        if not boxes:
            return
        
        print("\nRemoving parent boxes, keeping children...")
        filtered_bboxes = remove_parent_boxes(boxes, threshold=10)
        print(f"After filtering: {len(filtered_bboxes)} speech bubbles")
        
        # Create image with all bubble interiors colored blue
        print("\n" + "=" * 50)
        print("Coloring bubble interiors blue...")
        print("=" * 50)
        blue_image = color_bubble_interiors_blue(
            INPUT_IMAGE_PATH,
            filtered_bboxes,
            threshold_value=200
        )
        
        # Convert BGR numpy array to RGB PIL Image
        blue_image_rgb = cv2.cvtColor(blue_image, cv2.COLOR_BGR2RGB)
        blue_image_pil = Image.fromarray(blue_image_rgb)
        save_pil_image(blue_image_pil, BUBBLE_INTERIORS_BLUE_PATH)
        
        # Crop individual speech bubbles
        print("\nCropping speech bubbles...")
        cropped_images = get_cropped_images(INPUT_IMAGE_PATH, filtered_bboxes)
        print(f"Cropped {len(cropped_images)} speech bubble images")
        
        # Run OCR on each speech bubble and translate
        print("\n" + "=" * 50)
        print("Running OCR on each speech bubble...")
        print("=" * 50)
        
        bubble_texts = []  # List of (bbox, translated_text) tuples
        japanese_bboxes = []  # Track which bubbles contain Japanese
        
        for i, (cropped_img, bbox) in enumerate(cropped_images, 1):
            print(f"\n--- Speech Bubble {i} ---")
            translated_text = ""
            translation_success = False
            try:
                extracted_text = extract_text(cropped_img)
                sys.stdout.reconfigure(encoding='utf-8')
                print(f"Original text: {extracted_text}")
                
                # Translate the extracted text (includes CJK check)
                if extracted_text.strip():
                    try:
                        translated_text, translation_success = translate_phrase(extracted_text)
                        if translation_success:
                            print(f"Translated: {translated_text}")
                        else:
                            print(f"  Skipping bubble {i}: No Japanese text detected")
                    except Exception as e:
                        print(f"Error translating bubble {i}: {e}")
            except Exception as e:
                print(f"Error extracting text from bubble {i}: {e}")
            
            # Only process bubbles with successful translation (has CJK)
            if translation_success:
                japanese_bboxes.append(bbox)
                if translated_text:
                    bubble_texts.append((bbox, translated_text))
        
        # Clear speech bubbles before drawing translated text
        if not bubble_texts:
            return
        
        print("\n" + "=" * 50)
        print("Clearing speech bubbles...")
        print("=" * 50)
        
        # Generate bubble masks for drawing (before cleaning so we can use the original image)
        print("\n" + "=" * 50)
        print("Generating bubble masks...")
        print("=" * 50)
        
        # Load image for mask generation
        img_array = cv2.imread(INPUT_IMAGE_PATH)
        if img_array is None:
            raise ValueError(f"Could not load image from {INPUT_IMAGE_PATH}")
        
        # Generate masks for all bubbles with translations
        bubble_masks = {}
        for bbox, _ in bubble_texts:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Ensure bubble box is within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_array.shape[1], x2)
            y2 = min(img_array.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Crop the bubble
            bubble_crop = img_array[y1:y2, x1:x2].copy()
            
            # Get bubble interior mask
            bubble_mask, _ = get_bubble_text_mask(bubble_crop, threshold_value=200)
            bubble_masks[tuple(bbox)] = bubble_mask
        
        # Fill bubble interiors with base color (only for bubbles with Japanese text)
        cleaned_image = fill_bubble_interiors(
            INPUT_IMAGE_PATH, 
            japanese_bboxes, 
            threshold_value=200, 
            use_inpaint=False
        )
        
        # Convert BGR numpy array to RGB PIL Image
        cleaned_image_rgb = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
        cleaned_image_pil = Image.fromarray(cleaned_image_rgb)
        
        # Save cleaned image temporarily for draw_text_on_image
        save_pil_image(cleaned_image_pil, CLEANED_IMAGE_PATH)
        
        # Draw translated text on cleaned image
        print("\n" + "=" * 50)
        print("Drawing translated text on image...")
        print("=" * 50)
        
        translated_image = draw_text_on_image(
            CLEANED_IMAGE_PATH,
            bubble_texts,  # List of (bbox, translated_text) tuples
            bubble_masks,  # Dictionary of bbox -> mask
            font_path=FONT_PATH
        )
        save_pil_image(translated_image, TRANSLATED_OUTPUT_PATH)

    except Exception as e:
        print(f"Error in bubble detection: {e}")
    
    print("\n" + "=" * 50)
    print("Processing complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
