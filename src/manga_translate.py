"""
Comprehensive manga page translation pipeline.

This module provides a reusable function for translating manga pages
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import sys
import cv2
from PIL import Image

# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.webp', '.WEBP'}

from src.ocr import extract_text
from src.bubble import load_model, run_detection, process_detection_results
from src.image_utils import get_cropped_images, save_pil_image
from src.bbox import remove_parent_boxes, combine_overlapping_bubbles
from src.translate import translate_phrase
from src.draw_text import draw_text_on_image
from src.bubble_clean import fill_bubble_interiors, color_bubble_interiors_blue, get_bubble_text_mask


def translate_manga_page(
    input_image_path: str,
    output_folder: str,
    # Detection parameters
    detection_model=None,  # Optional: pass pre-loaded model or None to load
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    parent_box_threshold: int = 10,
    bbox_processing: str = 'remove-parent',
    # Bubble processing parameters
    threshold_value: int = 200,
    # Text parameters
    font_path: Optional[str] = None,
    # Translation parameters
    translation_model_path: Optional[str] = None,
    translation_device: str = 'cpu',
    translation_beam_size: int = 5,
    # OCR parameters
    ocr_model_id: str = "jzhang533/PaddleOCR-VL-For-Manga",
    ocr_max_new_tokens: int = 2048,
    # Output toggles
    save_speech_bubbles: bool = False,
    save_bubble_interiors: bool = False,
    save_cleaned: bool = False,
    save_translated: bool = True,
    # Filename customization
    output_filename: Optional[str] = None,  # If None, use input filename
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive function to translate a manga page from Japanese to English.
    
    This function performs the complete pipeline:
    1. Detects speech bubbles using YOLO
    2. Filters parent boxes
    3. Extracts text from each bubble using OCR
    4. Translates Japanese text to English
    5. Cleans bubble interiors
    6. Draws translated text on the image
    
    Args:
        input_image_path: Path to input manga page image
        output_folder: Base folder path for all outputs
        detection_model: Pre-loaded YOLO model (None to load automatically)
        conf_threshold: Confidence threshold for bubble detection (0-1)
        iou_threshold: IoU threshold for NMS (0-1)
        parent_box_threshold: Threshold for processing compound speech bubbles
        bbox_processing: Compound speech bubble processing mode ('remove-parent', 'combine-children', or 'none')
        threshold_value: Threshold for bubble mask detection
        font_path: Path to font file (None to use system default)
        translation_model_path: Path to translation model (None for default)
        translation_device: Device for translation ('cpu' or 'cuda')
        translation_beam_size: Beam size for translation
        ocr_model_id: Hugging Face model ID for OCR
        ocr_max_new_tokens: Maximum tokens for OCR generation
        save_speech_bubbles: If True, save annotated detection image
        save_bubble_interiors: If True, save blue bubble interior visualization
        save_cleaned: If True, save cleaned image (before text drawing)
        save_translated: If True, save final translated image
        output_filename: Custom output filename (None to derive from input)
        verbose: If True, print progress messages
    
    Returns:
        Dictionary containing:
        - 'annotated_image': PIL Image with detected bubbles (or None)
        - 'boxes': List of detected bounding boxes
        - 'filtered_boxes': List after removing parent boxes
        - 'blue_image': PIL Image with blue bubble interiors (or None)
        - 'cleaned_image': PIL Image with cleaned bubbles (or None)
        - 'translated_image': PIL Image with translated text (or None)
        - 'bubble_texts': List of (bbox, translated_text) tuples
        - 'bubble_masks': Dictionary mapping bbox to mask arrays
        - 'output_paths': Dictionary with saved file paths
    """
    # Initialize result dictionary
    results: Dict[str, Any] = {
        'annotated_image': None,
        'boxes': [],
        'filtered_boxes': [],
        'blue_image': None,
        'cleaned_image': None,
        'translated_image': None,
        'bubble_texts': [],
        'bubble_masks': {},
        'output_paths': {}
    }
    
    # Determine output filename
    if output_filename is None:
        input_path = Path(input_image_path)
        output_filename = input_path.stem
    
    # Create output folder structure
    output_base = Path(output_folder)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Create subfolders for each output type
    if save_speech_bubbles:
        (output_base / "speech_bubbles").mkdir(parents=True, exist_ok=True)
    if save_bubble_interiors:
        (output_base / "bubble_interiors").mkdir(parents=True, exist_ok=True)
    if save_cleaned:
        (output_base / "cleaned").mkdir(parents=True, exist_ok=True)
    if save_translated:
        (output_base / "translated").mkdir(parents=True, exist_ok=True)
    
    try:
        # Load detection model
        if detection_model is None:
            if verbose:
                print("\n" + "=" * 50)
                print("Loading detection model...")
                print("=" * 50)
            detection_model = load_model()
        
        # Detect speech bubbles
        if verbose:
            print("\n" + "=" * 50)
            print("Running speech bubble detection...")
            print("=" * 50)
        
        detection_results = run_detection(
            detection_model, 
            input_image_path, 
            conf_threshold=conf_threshold, 
            iou_threshold=iou_threshold
        )
        
        annotated_img, boxes = process_detection_results(detection_results)
        results['annotated_image'] = annotated_img
        results['boxes'] = boxes
        
        if verbose and annotated_img:
            print(f"\nTotal detections: {len(boxes)} speech bubbles found")
        
        # Save annotated image if requested
        if save_speech_bubbles and annotated_img:
            speech_bubbles_path = output_base / "speech_bubbles" / f"{output_filename}_speech_bubbles.png"
            save_pil_image(annotated_img, str(speech_bubbles_path), print_message=verbose)
            results['output_paths']['speech_bubbles'] = str(speech_bubbles_path)
        
        # Process bounding boxes based on mode
        if not boxes:
            if verbose:
                print("No speech bubbles detected.")
            return results
        
        if bbox_processing == 'remove-parent':
            if verbose:
                print("\nProcessing compound speech bubbles: removing parent boxes, keeping only children...")
            filtered_bboxes = remove_parent_boxes(boxes, threshold=parent_box_threshold)
        elif bbox_processing == 'combine-children':
            if verbose:
                print("\nProcessing compound speech bubbles: combining overlapping/touching bubbles...")
            filtered_bboxes = combine_overlapping_bubbles(boxes, touch_threshold=parent_box_threshold)
        else:  # 'none'
            if verbose:
                print("\nNo compound speech bubble processing applied...")
            filtered_bboxes = boxes
        
        results['filtered_boxes'] = filtered_bboxes
        if verbose:
            print(f"After processing: {len(filtered_bboxes)} speech bubbles")
        
        # Create image with all bubble interiors colored blue (if requested)
        if save_bubble_interiors:
            if verbose:
                print("\n" + "=" * 50)
                print("Coloring bubble interiors blue...")
                print("=" * 50)
            blue_image = color_bubble_interiors_blue(
                input_image_path,
                filtered_bboxes,
                threshold_value=threshold_value
            )
            # Convert BGR numpy array to RGB PIL Image
            blue_image_rgb = cv2.cvtColor(blue_image, cv2.COLOR_BGR2RGB)
            blue_image_pil = Image.fromarray(blue_image_rgb)
            results['blue_image'] = blue_image_pil
            
            bubble_interiors_path = output_base / "bubble_interiors" / f"{output_filename}_bubble_interiors.png"
            save_pil_image(blue_image_pil, str(bubble_interiors_path), print_message=verbose)
            results['output_paths']['bubble_interiors'] = str(bubble_interiors_path)
        
        # Crop individual speech bubbles
        if verbose:
            print("\nCropping speech bubbles...")
        cropped_images = get_cropped_images(input_image_path, filtered_bboxes)
        if verbose:
            print(f"Cropped {len(cropped_images)} speech bubble images")
        
        # Run OCR on each speech bubble and translate
        if verbose:
            print("\n" + "=" * 50)
            print("Running OCR on each speech bubble...")
            print("=" * 50)
        
        bubble_texts = []  # List of (bbox, translated_text) tuples
        japanese_bboxes = []  # Track which bubbles contain Japanese
        
        for i, (cropped_img, bbox) in enumerate(cropped_images, 1):
            if verbose:
                print(f"\n--- Speech Bubble {i} ---")
            translated_text = ""
            translation_success = False
            try:
                extracted_text = extract_text(cropped_img, model_id=ocr_model_id, max_new_tokens=ocr_max_new_tokens)
                if verbose:
                    sys.stdout.reconfigure(encoding='utf-8')
                    print(f"Original text: {extracted_text}")
                
                # Translate the extracted text (includes CJK check)
                if extracted_text.strip():
                    try:
                        translated_text, translation_success = translate_phrase(
                            extracted_text,
                            model_path=translation_model_path,
                            device=translation_device,
                            beam_size=translation_beam_size
                        )
                        if translation_success:
                            if verbose:
                                print(f"Translated: {translated_text}")
                        else:
                            if verbose:
                                print(f"  Skipping bubble {i}: No Japanese text detected")
                    except Exception as e:
                        if verbose:
                            print(f"Error translating bubble {i}: {e}")
            except Exception as e:
                if verbose:
                    print(f"Error extracting text from bubble {i}: {e}")
            
            # Only process bubbles with successful translation (has CJK)
            if translation_success:
                japanese_bboxes.append(bbox)
                if translated_text:
                    bubble_texts.append((bbox, translated_text))
        
        results['bubble_texts'] = bubble_texts
        
        # Clear speech bubbles before drawing translated text
        if not bubble_texts:
            if verbose:
                print("\nNo Japanese text found in any bubbles.")
            return results
        
        if verbose:
            print("\n" + "=" * 50)
            print("Clearing speech bubbles...")
            print("=" * 50)
        
        # Generate bubble masks for drawing (before cleaning so we can use the original image)
        if verbose:
            print("\n" + "=" * 50)
            print("Generating bubble masks...")
            print("=" * 50)
        
        # Load image for mask generation
        img_array = cv2.imread(input_image_path)
        if img_array is None:
            raise ValueError(f"Could not load image from {input_image_path}")
        
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
            bubble_mask, _ = get_bubble_text_mask(bubble_crop, threshold_value=threshold_value)
            bubble_masks[tuple(bbox)] = bubble_mask
        
        results['bubble_masks'] = bubble_masks
        
        # Fill bubble interiors with base color (only for bubbles with Japanese text)
        cleaned_image = fill_bubble_interiors(
            input_image_path, 
            japanese_bboxes, 
            threshold_value=threshold_value
        )
        
        # Convert BGR numpy array to RGB PIL Image
        cleaned_image_rgb = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
        cleaned_image_pil = Image.fromarray(cleaned_image_rgb)
        results['cleaned_image'] = cleaned_image_pil
        
        # Save cleaned image if requested, or create temporary file for draw_text_on_image
        # (draw_text_on_image needs a file path)
        if save_cleaned:
            cleaned_path = output_base / "cleaned" / f"{output_filename}_cleaned.png"
            save_pil_image(cleaned_image_pil, str(cleaned_path), print_message=verbose)
            results['output_paths']['cleaned'] = str(cleaned_path)
            cleaned_image_path = cleaned_path
        else:
            # Create temporary cleaned image for draw_text_on_image
            temp_cleaned_path = output_base / f"{output_filename}_temp_cleaned.png"
            save_pil_image(cleaned_image_pil, str(temp_cleaned_path), print_message=False)
            cleaned_image_path = temp_cleaned_path
        
        # Draw translated text on cleaned image
        if verbose:
            print("\n" + "=" * 50)
            print("Drawing translated text on image...")
            print("=" * 50)
        
        translated_image = draw_text_on_image(
            str(cleaned_image_path),
            bubble_texts,  # List of (bbox, translated_text) tuples
            bubble_masks,  # Dictionary of bbox -> mask
            font_path=font_path
        )
        results['translated_image'] = translated_image
        
        # Clean up temporary file if we created one
        if not save_cleaned:
            cleaned_image_path.unlink(missing_ok=True)
        
        # Save translated image if requested
        if save_translated:
            translated_path = output_base / "translated" / f"{output_filename}_translated.png"
            save_pil_image(translated_image, str(translated_path), print_message=verbose)
            results['output_paths']['translated'] = str(translated_path)
        
        if verbose:
            print("\n" + "=" * 50)
            print("Processing complete!")
            print("=" * 50)
    
    except Exception as e:
        if verbose:
            print(f"Error in manga translation: {e}")
        raise
    
    return results


def translate_manga_folder(
    input_folder: str,
    output_folder: str,
    # Detection parameters
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    parent_box_threshold: int = 10,
    bbox_processing: str = 'remove-parent',
    # Bubble processing parameters
    threshold_value: int = 200,
    # Text parameters
    font_path: Optional[str] = None,
    # Translation parameters
    translation_model_path: Optional[str] = None,
    translation_device: str = 'cpu',
    translation_beam_size: int = 5,
    # OCR parameters
    ocr_model_id: str = "jzhang533/PaddleOCR-VL-For-Manga",
    ocr_max_new_tokens: int = 2048,
    # Output toggles
    save_speech_bubbles: bool = False,
    save_bubble_interiors: bool = False,
    save_cleaned: bool = False,
    save_translated: bool = True,
    # Processing options
    verbose: bool = True,
    continue_on_error: bool = True
) -> Dict[str, Any]:
    """
    Translate all manga pages in a folder from Japanese to English.
    
    Processes all supported image files in the input folder, reusing the detection
    model across pages for efficiency. Each page is processed using translate_manga_page.
    
    Args:
        input_folder: Path to folder containing manga page images
        output_folder: Base folder path for all outputs (same structure as translate_manga_page)
        conf_threshold: Confidence threshold for bubble detection (0-1)
        iou_threshold: IoU threshold for NMS (0-1)
        parent_box_threshold: Threshold for processing compound speech bubbles
        bbox_processing: Compound speech bubble processing mode ('remove-parent', 'combine-children', or 'none')
        threshold_value: Threshold for bubble mask detection
        font_path: Path to font file (None to use system default)
        translation_model_path: Path to translation model (None for default)
        translation_device: Device for translation ('cpu' or 'cuda')
        translation_beam_size: Beam size for translation
        ocr_model_id: Hugging Face model ID for OCR
        ocr_max_new_tokens: Maximum tokens for OCR generation
        save_speech_bubbles: If True, save annotated detection images
        save_bubble_interiors: If True, save blue bubble interior visualizations
        save_cleaned: If True, save cleaned images (before text drawing)
        save_translated: If True, save final translated images
        verbose: If True, print progress messages
        continue_on_error: If True, continue processing other pages if one fails
    
    Returns:
        Dictionary containing:
        - 'processed_files': List of successfully processed file paths
        - 'failed_files': List of tuples (file_path, error_message) for failed files
        - 'results_by_file': Dictionary mapping file paths to their translate_manga_page results
        - 'total_files': Total number of image files found
        - 'successful_count': Number of successfully processed files
        - 'failed_count': Number of failed files
    """
    input_path = Path(input_folder)
    if not input_path.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_folder}")
    
    # Find all image files
    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix in SUPPORTED_IMAGE_EXTENSIONS
    ]
    
    # Sort files for consistent processing order
    image_files.sort(key=lambda x: x.name)
    
    total_files = len(image_files)
    
    if total_files == 0:
        if verbose:
            print(f"No image files found in {input_folder}")
        return {
            'processed_files': [],
            'failed_files': [],
            'results_by_file': {},
            'total_files': 0,
            'successful_count': 0,
            'failed_count': 0
        }
    
    if verbose:
        print("\n" + "=" * 50)
        print(f"Found {total_files} image file(s) to process")
        print("=" * 50)
    
    # Load detection model once for reuse across all pages
    detection_model = None
    if verbose:
        print("\nLoading detection model (will be reused for all pages)...")
    try:
        from src.bubble import load_model
        detection_model = load_model()
    except Exception as e:
        if not continue_on_error:
            raise
        if verbose:
            print(f"Warning: Could not load detection model: {e}")
    
    # Process each image file
    processed_files = []
    failed_files = []
    results_by_file = {}
    
    for i, image_file in enumerate(image_files, 1):
        if verbose:
            print("\n" + "=" * 50)
            print(f"Processing file {i}/{total_files}: {image_file.name}")
            print("=" * 50)
        
        try:
            # Use translate_manga_page with pre-loaded model
            result = translate_manga_page(
                input_image_path=str(image_file),
                output_folder=output_folder,
                detection_model=detection_model,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                parent_box_threshold=parent_box_threshold,
                bbox_processing=bbox_processing,
                threshold_value=threshold_value,
                font_path=font_path,
                translation_model_path=translation_model_path,
                translation_device=translation_device,
                translation_beam_size=translation_beam_size,
                ocr_model_id=ocr_model_id,
                ocr_max_new_tokens=ocr_max_new_tokens,
                save_speech_bubbles=save_speech_bubbles,
                save_bubble_interiors=save_bubble_interiors,
                save_cleaned=save_cleaned,
                save_translated=save_translated,
                output_filename=None,  # Use input filename
                verbose=verbose
            )
            
            processed_files.append(str(image_file))
            results_by_file[str(image_file)] = result
            
        except Exception as e:
            error_msg = str(e)
            failed_files.append((str(image_file), error_msg))
            
            if verbose:
                print(f"\nError processing {image_file.name}: {error_msg}")
            
            if not continue_on_error:
                raise
    
    # Print summary
    if verbose:
        print("\n" + "=" * 50)
        print("Folder processing complete!")
        print("=" * 50)
        print(f"Total files: {total_files}")
        print(f"Successfully processed: {len(processed_files)}")
        print(f"Failed: {len(failed_files)}")
        if failed_files:
            print("\nFailed files:")
            for file_path, error in failed_files:
                print(f"  - {Path(file_path).name}: {error}")
    
    return {
        'processed_files': processed_files,
        'failed_files': failed_files,
        'results_by_file': results_by_file,
        'total_files': total_files,
        'successful_count': len(processed_files),
        'failed_count': len(failed_files)
    }

