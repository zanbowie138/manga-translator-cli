from paddle import extract_text, detect_layout, detect_text_regions
from bubble import predict_image, get_detections, crop_and_save_bubbles, combine_overlapping_bubbles
from pathlib import Path
import sys

def main():
    """Main function that orchestrates both PaddleOCR and bubble detection"""
    image_path = "image_15.png"
    
    print("=" * 50)
    print("Running PaddleOCR text extraction...")
    print("=" * 50)
    
    # Extract text using PaddleOCR-VL
    try:
        extracted_text = extract_text(image_path)
        print("-" * 30)
        print("Extracted Text:")
        sys.stdout.reconfigure(encoding='utf-8')
        print(extracted_text)
        print("-" * 30)
    except Exception as e:
        print(f"Error in text extraction: {e}")
    
    print("\n" + "=" * 50)
    print("Running speech bubble detection...")
    print("=" * 50)
    
    # Detect speech bubbles
    try:
        results, annotated_img = predict_image(
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
                saved_paths = crop_and_save_bubbles(
                    image_path,
                    merged_detections,
                    output_dir="output",
                    prefix="bubble"
                )
                print(f"Saved {len(saved_paths)} cropped speech bubble images")
            
            # Display results
            try:
                annotated_img.show()
            except:
                print("Could not display image (may require GUI)")
    except Exception as e:
        print(f"Error in bubble detection: {e}")
    
    print("\n" + "=" * 50)
    print("Processing complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
