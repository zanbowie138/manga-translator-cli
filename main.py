from textocr import extract_text
from bubble import predict_image, get_detections, crop_and_save_bubbles, combine_overlapping_bubbles
from translate import translate_phrase
from draw_text import draw_translated_text_on_image
from bubble_clean import fill_bubble_interiors
from pathlib import Path
import sys
import cv2
from PIL import Image   

def main():
    """Main function that orchestrates both PaddleOCR and bubble detection"""
    image_path = "atelier.png"
    
    print("\n" + "=" * 50)
    print("Running speech bubble detection...")
    print("=" * 50)
    
    # Detect speech bubbles
    try:
        _, annotated_img = predict_image(
            image_path,
            conf_threshold=0.25,
            iou_threshold=0.45,
            save_output=True,
            output_dir="output"
        )
        
        if annotated_img:
            # Save to custom location
            output_path = "output/speech_bubbles.png"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            annotated_img.save(output_path)
            print(f"Annotated image saved to: {output_path}")
            
            # Get structured detections
            detections = get_detections(image_path, conf_threshold=0.25)
            print(f"\nTotal detections: {len(detections)} speech bubbles found")
            
            # Combine overlapping/touching/nested bubbles
            if detections:
                print("\nCombining overlapping and touching speech bubbles...")
                merged_detections = combine_overlapping_bubbles(detections, touch_threshold=10)
                print(f"After merging: {len(merged_detections)} speech bubbles")
                
                # Crop and save individual speech bubbles
                print("\nCropping and saving speech bubbles...")
                saved_paths, cropped_images = crop_and_save_bubbles(
                    image_path,
                    merged_detections,
                    output_dir="output",
                    prefix="bubble"
                )
                print(f"Saved {len(saved_paths)} cropped speech bubble images")
                
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
                        image_path, 
                        bubble_boxes, 
                        threshold_value=200, 
                        use_inpaint=False
                    )
                    
                    # Convert BGR numpy array to RGB PIL Image
                    cleaned_image_rgb = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
                    cleaned_image_pil = Image.fromarray(cleaned_image_rgb)
                    
                    # Save cleaned image temporarily for draw_translated_text_on_image
                    cleaned_image_path = "output/cleaned_for_translation.png"
                    cleaned_image_pil.save(cleaned_image_path)
                    print(f"Cleaned image saved to: {cleaned_image_path}")
                    
                    # Draw translated text on cleaned image
                    print("\n" + "=" * 50)
                    print("Drawing translated text on image...")
                    print("=" * 50)
                    
                    translated_output_path = "output/image_with_translations.png"
                    translated_image = draw_translated_text_on_image(
                        cleaned_image_path,
                        detections_translations,
                        translated_output_path
                    )
                    print(f"Image with translations saved to: {translated_output_path}")

    except Exception as e:
        print(f"Error in bubble detection: {e}")
    
    print("\n" + "=" * 50)
    print("Processing complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
