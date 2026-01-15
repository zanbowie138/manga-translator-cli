from src.manga_ocr import extract_text
from src.bubble import find_speech_bubbles, get_detections, get_cropped_images
from src.bbox import combine_overlapping_bubbles
from src.translate import translate_phrase
from src.draw_text import draw_translated_text_on_image
from src.bubble_clean import fill_bubble_interiors
from pathlib import Path
import sys
import cv2
from PIL import Image

# Image path constants
INPUT_IMAGE_PATH = "input/image_15.png"
OUTPUT_DIR = "output"
SPEECH_BUBBLES_OUTPUT_PATH = "output/speech_bubbles.png"
CLEANED_IMAGE_PATH = "output/cleaned.png"
TRANSLATED_OUTPUT_PATH = "output/translated.png"

def main():
    """Main function that orchestrates both PaddleOCR and bubble detection"""
    
    print("\n" + "=" * 50)
    print("Running speech bubble detection...")
    print("=" * 50)
    
    # Detect speech bubbles
    try:
        _, annotated_img = find_speech_bubbles(
            INPUT_IMAGE_PATH,
            conf_threshold=0.25,
            iou_threshold=0.45,
            save_output=True,
            output_dir=OUTPUT_DIR
        )
        
        if annotated_img:
            # Save to custom location
            Path(SPEECH_BUBBLES_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
            annotated_img.save(SPEECH_BUBBLES_OUTPUT_PATH)
            print(f"Annotated image saved to: {SPEECH_BUBBLES_OUTPUT_PATH}")
            
            # Get structured detections
            detections = get_detections(INPUT_IMAGE_PATH, conf_threshold=0.25)
            print(f"\nTotal detections: {len(detections)} speech bubbles found")
            
            # Combine overlapping/touching/nested bubbles
            if detections:
                print("\nCombining overlapping and touching speech bubbles...")
                merged_detections = combine_overlapping_bubbles(detections, touch_threshold=10)
                print(f"After merging: {len(merged_detections)} speech bubbles")
                
                # Crop individual speech bubbles
                print("\nCropping speech bubbles...")
                cropped_images = get_cropped_images(INPUT_IMAGE_PATH, merged_detections)
                print(f"Cropped {len(cropped_images)} speech bubble images")
                
                # Run OCR on each merged speech bubble and translate
                print("\n" + "=" * 50)
                print("Running OCR on each speech bubble...")
                print("=" * 50)
                
                detections_translations = []
                
                for i, (cropped_img, det) in enumerate(cropped_images, 1):
                    print(f"\n--- Speech Bubble {i} (confidence: {det['confidence']:.3f}) ---")
                    translated_text = ""
                    try:
                        extracted_text = extract_text(cropped_img)
                        sys.stdout.reconfigure(encoding='utf-8')
                        print(f"Original text: {extracted_text}")
                        
                        # Translate the extracted text
                        if extracted_text.strip():
                            try:
                                translated_text = translate_phrase(extracted_text)
                                print(f"Translated: {translated_text}")
                            except Exception as e:
                                print(f"Error translating bubble {i}: {e}")
                    except Exception as e:
                        print(f"Error extracting text from bubble {i}: {e}")
                    
                    # Store detection and translation for drawing
                    if translated_text:
                        detections_translations.append((det, translated_text))
                
                # Clear speech bubbles before drawing translated text
                if detections_translations:
                    print("\n" + "=" * 50)
                    print("Clearing speech bubbles...")
                    print("=" * 50)
                    
                    # Extract bubble bounding boxes
                    bubble_boxes = [det['bbox'] for det in merged_detections]
                    
                    # Fill bubble interiors with base color
                    cleaned_image = fill_bubble_interiors(
                        INPUT_IMAGE_PATH, 
                        bubble_boxes, 
                        threshold_value=200, 
                        use_inpaint=False
                    )
                    
                    # Convert BGR numpy array to RGB PIL Image
                    cleaned_image_rgb = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
                    cleaned_image_pil = Image.fromarray(cleaned_image_rgb)
                    
                    # Save cleaned image temporarily for draw_translated_text_on_image
                    cleaned_image_pil.save(CLEANED_IMAGE_PATH)
                    print(f"Cleaned image saved to: {CLEANED_IMAGE_PATH}")
                    
                    # Draw translated text on cleaned image
                    print("\n" + "=" * 50)
                    print("Drawing translated text on image...")
                    print("=" * 50)
                    
                    translated_image = draw_translated_text_on_image(
                        CLEANED_IMAGE_PATH,
                        detections_translations,
                        TRANSLATED_OUTPUT_PATH
                    )
                    print(f"Image with translations saved to: {TRANSLATED_OUTPUT_PATH}")

    except Exception as e:
        print(f"Error in bubble detection: {e}")
    
    print("\n" + "=" * 50)
    print("Processing complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
