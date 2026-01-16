"""
Comprehensive manga page translation pipeline.

This module provides a reusable function for translating manga pages
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import sys
import numpy as np
import cv2
from PIL import Image

# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.webp', '.WEBP'}

from src.ocr import extract_text
from src.bubble_detect import load_bubble_detection_model, run_detection
from src.image_utils import get_cropped_images, save_pil_image
from src.bbox import BoundingBox, remove_parent_boxes, combine_overlapping_bubbles
from src.translate import translate_phrase
from src.draw_text import draw_text_on_image
from src.bubble_clean import fill_bubble_interiors, visualize_bubble_masks, visualize_single_bubble_mask, get_bubble_text_mask


def _setup_output_structure(
    output_folder: str,
    output_filename: str,
    output_subfolder: Optional[str],
    save_speech_bubbles: bool,
    save_bubble_interiors: bool,
    save_cleaned: bool,
    save_translated: bool
) -> Path:
    """
    Set up output folder structure and return output base path.
    
    Args:
        output_folder: Base folder path for all outputs
        output_filename: Output filename (without extension)
        output_subfolder: Optional subfolder name
        save_speech_bubbles: Whether to create speech_bubbles folder
        save_bubble_interiors: Whether to create bubble_interiors folder
        save_cleaned: Whether to create cleaned folder
        save_translated: Whether to create translated folder
    
    Returns:
        Path to output base directory
    """
    output_base = Path(output_folder)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Create subfolders for each output type
    if save_speech_bubbles:
        _get_output_dir_path(output_base, "speech_bubbles", output_subfolder).mkdir(parents=True, exist_ok=True)
    if save_bubble_interiors:
        _get_output_dir_path(output_base, "bubble_interiors", output_subfolder).mkdir(parents=True, exist_ok=True)
    if save_cleaned:
        _get_output_dir_path(output_base, "cleaned", output_subfolder).mkdir(parents=True, exist_ok=True)
    if save_translated:
        _get_output_dir_path(output_base, "translated", output_subfolder).mkdir(parents=True, exist_ok=True)
    
    return output_base


def _detect_speech_bubbles(
    input_image: Image.Image,
    detection_model: Any,
    conf_threshold: float,
    iou_threshold: float,
    save_speech_bubbles: bool,
    output_base: Path,
    output_filename: str,
    output_subfolder: Optional[str],
    silent: bool = False
) -> Tuple[Optional[Image.Image], List[BoundingBox], Dict[str, Any]]:
    """
    Detect speech bubbles in the image.
    
    Args:
        input_image: Input PIL Image
        detection_model: Pre-loaded YOLO model
        conf_threshold: Confidence threshold for detection
        iou_threshold: IoU threshold for NMS
        save_speech_bubbles: Whether to save annotated image
        output_base: Base output directory
        output_filename: Output filename
        output_subfolder: Optional subfolder name
        silent: If True, suppress progress messages
    
    Returns:
        Tuple of (annotated_image, boxes, output_paths_dict)
    """
    if not silent:
        print("Running speech bubble detection...")
    
    annotated_bubble_img, boxes = run_detection(
        detection_model,
        input_image,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        silent=silent
    )
    
    output_paths = {}
    
    if not silent and annotated_bubble_img:
        print(f"Total detections: {len(boxes)} speech bubbles found")
    
    # Save annotated image if requested
    if save_speech_bubbles and annotated_bubble_img:
        speech_bubbles_path = _get_output_file_path(
            output_base, "speech_bubbles", f"{output_filename}_speech_bubbles.png", output_subfolder
        )
        save_pil_image(annotated_bubble_img, str(speech_bubbles_path), print_message=not silent)
        output_paths['speech_bubbles'] = str(speech_bubbles_path)
    
    return annotated_bubble_img, boxes, output_paths


def _process_bounding_boxes(
    boxes: List[BoundingBox],
    bbox_processing: str,
    parent_box_threshold: float,
    silent: bool = False
) -> List[BoundingBox]:
    """
    Process bounding boxes based on the specified mode.
    
    Args:
        boxes: List of detected bounding boxes
        bbox_processing: Processing mode ('remove-parent', 'combine-children', or 'none')
        parent_box_threshold: Threshold for processing
        silent: If True, suppress progress messages
    
    Returns:
        List of processed bounding boxes
    """
    if not boxes:
        if not silent:
            print("No speech bubbles detected.")
        return []
    
    if bbox_processing == 'remove-parent':
        if not silent:
            print("\nProcessing compound speech bubbles: removing parent boxes, keeping only children...")
        filtered_bboxes = remove_parent_boxes(boxes, threshold=parent_box_threshold)
    elif bbox_processing == 'combine-children':
        if not silent:
            print("\nProcessing compound speech bubbles: combining overlapping/touching bubbles...")
        filtered_bboxes = combine_overlapping_bubbles(boxes, touch_threshold=parent_box_threshold)
    else:  # 'none'
        if not silent:
            print("\nNo compound speech bubble processing applied...")
        filtered_bboxes = boxes
    
    if not silent:
        print(f"After processing: {len(filtered_bboxes)} speech bubbles")
    
    return filtered_bboxes


def _create_bubble_interiors_visualization(
    input_image: Image.Image,
    bubble_texts: List[Tuple[BoundingBox, str]],
    bubble_masks: Dict[BoundingBox, np.ndarray],
    save_bubble_interiors: bool,
    output_base: Path,
    output_filename: str,
    output_subfolder: Optional[str],
    silent: bool = False
) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
    """
    Create visualization of bubble interiors using pre-generated bubble masks.
    
    Args:
        input_image: Input PIL Image
        bubble_texts: List of (bbox, translated_text) tuples
        bubble_masks: Dictionary mapping BoundingBox to mask arrays
        save_bubble_interiors: Whether to save visualization
        output_base: Base output directory
        output_filename: Output filename
        output_subfolder: Optional subfolder name
        silent: If True, suppress progress messages
    
    Returns:
        Tuple of (bubble_masks_image, output_paths_dict)
    """
    if not save_bubble_interiors:
        return None, {}
    
    if not silent:
        print("Creating bubble interiors visualization...")
    
    # Convert PIL Image to BGR numpy array
    img_array = np.array(input_image.convert("RGB"))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to BGRA (add alpha channel)
    output = cv2.cvtColor(img_array, cv2.COLOR_BGR2BGRA)
    
    # Define colors with transparency (BGR + Alpha)
    blue_color = np.array([255, 0, 0, 128], dtype=np.uint8)  # Transparent blue (BGR + Alpha)
    green_color = np.array([0, 255, 0, 128], dtype=np.uint8)  # Transparent green (BGR + Alpha)
    
    # Process each bubble using pre-generated masks
    for bbox, _ in bubble_texts:
        if bbox not in bubble_masks:
            continue
        
        # Clip to image bounds
        clipped_box = bbox.clip(img_array.shape[1], img_array.shape[0])
        if not clipped_box.is_valid():
            continue
        
        x1, y1, x2, y2 = clipped_box
        
        # Get the pre-generated bubble mask
        bubble_mask = bubble_masks[bbox]
        
        # Create a region overlay for this bubble
        bubble_region = output[y1:y2, x1:x2].copy()
        
        # Create mask for area outside bubble but within bounding box
        box_exterior_mask = 255 - bubble_mask
        
        # Fill bubble interior with transparent blue
        bubble_indices = bubble_mask > 0
        bubble_region[bubble_indices] = blue_color
        
        # Fill area outside bubble (but within bounding box) with transparent green
        exterior_indices = box_exterior_mask > 0
        bubble_region[exterior_indices] = green_color
        
        # Paste the visualized bubble back into the output image
        output[y1:y2, x1:x2] = bubble_region
    
    # Convert BGRA numpy array to RGBA PIL Image
    bubble_masks_rgba = cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)
    bubble_masks_pil = Image.fromarray(bubble_masks_rgba)
    
    output_paths = {}
    bubble_interiors_path = _get_output_file_path(
        output_base, "bubble_interiors", f"{output_filename}_bubble_interiors.png", output_subfolder
    )
    save_pil_image(bubble_masks_pil, str(bubble_interiors_path), print_message=not silent)
    output_paths['bubble_interiors'] = str(bubble_interiors_path)
    
    return bubble_masks_pil, output_paths


def _extract_and_translate_text(
    cropped_images: List[Tuple[Image.Image, BoundingBox]],
    ocr_model_id: str,
    ocr_max_new_tokens: int,
    translation_model_path: Optional[str],
    translation_device: str,
    translation_beam_size: int,
    silent: bool = False
) -> Tuple[List[Tuple[BoundingBox, str]], List[BoundingBox]]:
    """
    Extract text from each bubble using OCR and translate it.
    Processes extraction first for all bubbles, then batch translates all texts.
    
    Args:
        cropped_images: List of (cropped_image, bbox) tuples
        ocr_model_id: Hugging Face model ID for OCR
        ocr_max_new_tokens: Maximum tokens for OCR generation
        translation_model_path: Path to translation model
        translation_device: Device for translation
        translation_beam_size: Beam size for translation
        silent: If True, suppress progress messages
    
    Returns:
        Tuple of (bubble_texts, japanese_bboxes) where:
        - bubble_texts: List of (bbox, translated_text) tuples
        - japanese_bboxes: List of bounding boxes containing Japanese text
    """
    if not silent:
        print("\n" + "=" * 50)
        print("Running OCR on all speech bubbles...")
        print("=" * 50)
    
    # Step 1: Extract text from all bubbles
    extracted_texts = []  # List of (bbox, extracted_text, index) tuples
    for i, (cropped_img, bbox) in enumerate(cropped_images, 1):
        if not silent:
            print(f"\n--- Speech Bubble {i} ---")
            print(f"Bounding box: {bbox}")
        try:
            extracted_text = extract_text(cropped_img, model_id=ocr_model_id, max_new_tokens=ocr_max_new_tokens)
            if not silent:
                sys.stdout.reconfigure(encoding='utf-8')
                print(f"Original text: {extracted_text}")
            extracted_texts.append((bbox, extracted_text, i))
        except Exception as e:
            if not silent:
                print(f"Error extracting text from bubble {i}: {e}")
            extracted_texts.append((bbox, "", i))
    
    # Step 2: Filter texts with CJK characters and prepare for batch translation
    from src.translate import is_cjk
    
    texts_to_translate = []  # List of (bbox, extracted_text, index) tuples with CJK
    for bbox, extracted_text, index in extracted_texts:
        if extracted_text.strip():
            has_cjk = any(is_cjk(c) for c in extracted_text)
            if has_cjk:
                texts_to_translate.append((bbox, extracted_text, index))
    
    if not texts_to_translate:
        if not silent:
            print("\nNo Japanese text found in any bubbles.")
        return [], []
    
    # Step 3: Batch translate all texts
    if not silent:
        print("\n" + "=" * 50)
        print(f"Translating {len(texts_to_translate)} text(s) with Japanese characters...")
        print("=" * 50)
    
    # Import translation functions
    from src.translate import load_translation_models, is_cjk
    
    # Load translation models
    translator, tokenizer_source, tokenizer_target = load_translation_models(translation_model_path, translation_device)
    
    # Prepare all texts for batch translation
    tokenized_texts = []
    valid_indices = []
    for bbox, extracted_text, index in texts_to_translate:
        try:
            tokenized = tokenizer_source.encode(extracted_text, out_type=str)
            tokenized_texts.append(tokenized)
            valid_indices.append((bbox, extracted_text, index))
        except Exception as e:
            if not silent:
                print(f"Error tokenizing text from bubble {index}: {e}")
    
    if not tokenized_texts:
        return [], []
    
    # Batch translate
    try:
        translated_results = translator.translate_batch(
            source=tokenized_texts,
            beam_size=translation_beam_size
        )
    except Exception as e:
        if not silent:
            print(f"Error in batch translation: {e}")
        return [], []
    
    # Step 4: Decode translations and match back to bubbles
    bubble_texts = []  # List of (bbox, translated_text) tuples
    japanese_bboxes = []  # Track which bubbles contain Japanese
    
    for (bbox, extracted_text, index), translated_result in zip(valid_indices, translated_results):
        try:
            # TODO: Add functionality to manage multiple hypotheses
            # TODO: Add an option to replace the <unk> token with the most likely translation or a custom placeholder
            translated_text = tokenizer_target.decode(translated_result.hypotheses[0]).replace('<unk>', '')
            if translated_text.strip():
                bubble_texts.append((bbox, translated_text))
                japanese_bboxes.append(bbox)
                if not silent:
                    print(f"\nBubble {index}:")
                    print(f"  Original: {extracted_text}")
                    print(f"  Translated: {translated_text}")
        except Exception as e:
            if not silent:
                print(f"Error decoding translation for bubble {index}: {e}")
    
    return bubble_texts, japanese_bboxes


def _generate_bubble_masks(
    input_image: Image.Image,
    bubble_texts: List[Tuple[BoundingBox, str]],
    threshold_value: int,
    silent: bool = False
) -> Dict[BoundingBox, np.ndarray]:
    """
    Generate bubble masks for drawing translated text.
    
    Args:
        input_image: Input PIL Image
        bubble_texts: List of (bbox, translated_text) tuples
        threshold_value: Threshold for bubble mask detection
        silent: If True, suppress progress messages
    
    Returns:
        Dictionary mapping BoundingBox to mask arrays
    """
    if not silent:
        print("\n" + "=" * 50)
        print("Generating bubble masks...")
        print("=" * 50)
    
    # Convert PIL Image to BGR numpy array
    img_array = np.array(input_image.convert("RGB"))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Generate masks for all bubbles with translations
    bubble_masks = {}
    for bbox, _ in bubble_texts:
        # Clip bounding box to image bounds
        clipped_box = bbox.clip(img_array.shape[1], img_array.shape[0])
        
        # Validate that the box is valid
        if not clipped_box.is_valid():
            continue
        
        x1, y1, x2, y2 = clipped_box
        
        # Crop the bubble
        bubble_crop = img_array[y1:y2, x1:x2].copy()
        
        # Get bubble interior mask
        bubble_mask, _ = get_bubble_text_mask(bubble_crop, threshold_value=threshold_value)
        bubble_masks[bbox] = bubble_mask  # Use BoundingBox as key (it's hashable)
    
    return bubble_masks


def _clean_bubble_interiors(
    input_image: Image.Image,
    japanese_bboxes: List[BoundingBox],
    threshold_value: int,
    save_cleaned: bool,
    output_base: Path,
    output_filename: str,
    output_subfolder: Optional[str],
    silent: bool = False
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Fill bubble interiors with base color.
    
    Args:
        input_image: Input PIL Image
        japanese_bboxes: List of bounding boxes containing Japanese text
        threshold_value: Threshold for bubble mask detection
        save_cleaned: Whether to save cleaned image
        output_base: Base output directory
        output_filename: Output filename
        output_subfolder: Optional subfolder name
        silent: If True, suppress progress messages
    
    Returns:
        Tuple of (cleaned_image_pil, output_paths_dict)
    """
    if not silent:
        print("\n" + "=" * 50)
        print("Clearing speech bubbles...")
        print("=" * 50)
    
    # Fill bubble interiors with base color (only for bubbles with Japanese text)
    cleaned_image = fill_bubble_interiors(
        input_image,
        japanese_bboxes,
        threshold_value=threshold_value
    )
    
    # Convert BGR numpy array to RGB PIL Image
    cleaned_image_rgb = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
    cleaned_image_pil = Image.fromarray(cleaned_image_rgb)
    
    output_paths = {}
    
    # Save cleaned image if requested
    if save_cleaned:
        cleaned_path = _get_output_file_path(
            output_base, "cleaned", f"{output_filename}_cleaned.png", output_subfolder
        )
        save_pil_image(cleaned_image_pil, str(cleaned_path), print_message=not silent)
        output_paths['cleaned'] = str(cleaned_path)
    
    return cleaned_image_pil, output_paths


def _draw_translated_text(
    cleaned_image: Image.Image,
    bubble_texts: List[Tuple[BoundingBox, str]],
    bubble_masks: Dict[BoundingBox, np.ndarray],
    font_path: Optional[str],
    save_translated: bool,
    output_base: Path,
    output_filename: str,
    output_subfolder: Optional[str],
    silent: bool = False
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Draw translated text on the cleaned image.
    
    Args:
        cleaned_image: Cleaned PIL Image
        bubble_texts: List of (bbox, translated_text) tuples
        bubble_masks: Dictionary mapping BoundingBox to mask arrays
        font_path: Path to font file
        save_translated: Whether to save translated image
        output_base: Base output directory
        output_filename: Output filename
        output_subfolder: Optional subfolder name
        silent: If True, suppress progress messages
    
    Returns:
        Tuple of (translated_image, output_paths_dict)
    """
    if not silent:
        print("\n" + "=" * 50)
        print("Drawing translated text on image...")
        print("=" * 50)
    
    translated_image = draw_text_on_image(
        cleaned_image,
        bubble_texts,  # List of (bbox, translated_text) tuples
        bubble_masks,  # Dictionary of bbox -> mask
        font_path=font_path
    )
    
    output_paths = {}
    
    # Save translated image if requested
    if save_translated:
        translated_path = _get_output_file_path(
            output_base, "translated", f"{output_filename}_translated.png", output_subfolder
        )
        save_pil_image(translated_image, str(translated_path), print_message=not silent)
        output_paths['translated'] = str(translated_path)
    
    return translated_image, output_paths




def _get_output_dir_path(output_base: Path, folder_name: str, output_subfolder: Optional[str] = None) -> Path:
    """
    Get the output directory path, optionally nested under a subfolder.
    
    Args:
        output_base: Base output directory path
        folder_name: Name of the output folder (e.g., "translated", "cleaned")
        output_subfolder: Optional subfolder name to nest outputs under
    
    Returns:
        Path to the output directory
    """
    if output_subfolder:
        return output_base / folder_name / output_subfolder
    else:
        return output_base / folder_name


def _get_output_file_path(output_base: Path, folder_name: str, filename: str, output_subfolder: Optional[str] = None) -> Path:
    """
    Get the output file path, optionally nested under a subfolder.
    
    Args:
        output_base: Base output directory path
        folder_name: Name of the output folder (e.g., "translated", "cleaned")
        filename: Name of the output file
        output_subfolder: Optional subfolder name to nest outputs under
    
    Returns:
        Path to the output file
    """
    if output_subfolder:
        return output_base / folder_name / output_subfolder / filename
    else:
        return output_base / folder_name / filename


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
    output_subfolder: Optional[str] = None,  # Optional subfolder name (e.g., for folder processing)
    silent: bool = False
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
        output_subfolder: Optional subfolder name for organizing outputs (e.g., input folder name)
        verbose: If True, print progress messages
    
    Returns:
        Dictionary containing:
        - 'annotated_bubble_image': PIL Image with detected bubbles (or None)
        - 'boxes': List of detected BoundingBox instances
        - 'filtered_boxes': List of BoundingBox instances after removing parent boxes
        - 'blue_image': PIL Image with blue bubble interiors (or None)
        - 'cleaned_image': PIL Image with cleaned bubbles (or None)
        - 'translated_image': PIL Image with translated text (or None)
        - 'bubble_texts': List of (BoundingBox, translated_text) tuples
        - 'bubble_masks': Dictionary mapping BoundingBox to mask arrays
        - 'bubble_interiors_visualization': PIL Image with bubble interiors visualization (or None)
        - 'output_paths': Dictionary with saved file paths
    """
    # Initialize result dictionary
    results: Dict[str, Any] = {
        'annotated_bubble_image': None,
        'boxes': [],
        'filtered_boxes': [],
        'blue_image': None,
        'cleaned_image': None,
        'translated_image': None,
        'bubble_texts': [],
        'bubble_masks': {},  # Dictionary mapping BoundingBox to mask arrays
        'bubble_interiors_visualization': None,  # PIL Image visualization
        'output_paths': {}
    }
    
    # Determine output filename
    if output_filename is None:
        input_path = Path(input_image_path)
        output_filename = input_path.stem
    
    # Set up output folder structure
    output_base = _setup_output_structure(
        output_folder,
        output_filename,
        output_subfolder,
        save_speech_bubbles,
        save_bubble_interiors,
        save_cleaned,
        save_translated
    )
    
    try:
        # Load the input image
        input_image = Image.open(input_image_path).convert("RGB")
        
        # Load detection model
        if detection_model is None:
            if not silent:
                print("Loading detection model...")
            detection_model = load_bubble_detection_model()
        
        # Detect speech bubbles
        annotated_bubble_img, boxes, output_paths = _detect_speech_bubbles(
            input_image,
            detection_model, 
            conf_threshold,
            iou_threshold,
            save_speech_bubbles,
            output_base,
            output_filename,
            output_subfolder,
            silent
        )
        
        results['annotated_bubble_image'] = annotated_bubble_img
        results['boxes'] = boxes
        results['output_paths'].update(output_paths)
        
        # Process bounding boxes based on mode
        filtered_bboxes = _process_bounding_boxes(
            boxes,
            bbox_processing,
            parent_box_threshold,
            silent
        )
        
        results['filtered_boxes'] = filtered_bboxes
        
        if not filtered_bboxes:
            return results
        
        # Crop individual speech bubbles
        if not silent:
            print("\nCropping speech bubbles...")
        cropped_images = get_cropped_images(input_image, filtered_bboxes)
        if not silent:
            print(f"Cropped {len(cropped_images)} speech bubble images")
        
        # Extract and translate text
        bubble_texts, japanese_bboxes = _extract_and_translate_text(
            cropped_images,
            ocr_model_id,
            ocr_max_new_tokens,
            translation_model_path,
            translation_device,
            translation_beam_size,
            silent
        )
        
        results['bubble_texts'] = bubble_texts
        
        # Check if we have any Japanese text to translate
        if not bubble_texts:
            if not silent:
                print("\nNo Japanese text found in any bubbles.")
            return results
        
        # Generate bubble masks for drawing
        bubble_masks = _generate_bubble_masks(
            input_image,
            bubble_texts,
            threshold_value,
            silent
        )
        
        results['bubble_masks'] = bubble_masks
        
        # Create bubble interiors visualization using pre-generated masks
        bubble_interiors_pil, output_paths = _create_bubble_interiors_visualization(
            input_image,
            bubble_texts,
            bubble_masks,
            save_bubble_interiors,
            output_base,
            output_filename,
            output_subfolder,
            silent
        )
        
        if bubble_interiors_pil is not None:
            results['bubble_interiors_visualization'] = bubble_interiors_pil
        results['output_paths'].update(output_paths)
        
        # Clean bubble interiors
        cleaned_image_pil, output_paths = _clean_bubble_interiors(
            input_image,
            japanese_bboxes, 
            threshold_value,
            save_cleaned,
            output_base,
            output_filename,
            output_subfolder,
            silent
        )
        
        results['cleaned_image'] = cleaned_image_pil
        results['output_paths'].update(output_paths)
        
        # Draw translated text
        translated_image, output_paths = _draw_translated_text(
            cleaned_image_pil,
            bubble_texts,
            bubble_masks,
            font_path,
            save_translated,
            output_base,
            output_filename,
            output_subfolder,
            silent
        )
        
        results['translated_image'] = translated_image
        results['output_paths'].update(output_paths)
        
        if not silent:
            print("\n" + "=" * 50)
            print("Processing complete!")
            print("=" * 50)
    
    except Exception as e:
        if not silent:
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
    silent: bool = False,
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
    
    # Extract folder name for organizing outputs
    folder_name = input_path.name
    
    # Find all image files
    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix in SUPPORTED_IMAGE_EXTENSIONS
    ]
    
    # Sort files for consistent processing order
    image_files.sort(key=lambda x: x.name)
    
    total_files = len(image_files)
    
    if total_files == 0:
        if not silent:
            print(f"No image files found in {input_folder}")
        return {
            'processed_files': [],
            'failed_files': [],
            'results_by_file': {},
            'total_files': 0,
            'successful_count': 0,
            'failed_count': 0
        }
    
    if not silent:
        print("\n" + "=" * 50)
        print(f"Found {total_files} image file(s) to process")
        print("=" * 50)
    
    # Load detection model once for reuse across all pages
    detection_model = None
    if not silent:
        print("\nLoading detection model (will be reused for all pages)...")
    try:
        detection_model = load_bubble_detection_model()
    except Exception as e:
        if not continue_on_error:
            raise
        if not silent:
            print(f"Warning: Could not load detection model: {e}")
    
    # Process each image file
    processed_files = []
    failed_files = []
    results_by_file = {}
    
    for i, image_file in enumerate(image_files, 1):
        if not silent:
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
                output_subfolder=folder_name,  # Organize outputs by folder name
                silent=silent
            )
            
            processed_files.append(str(image_file))
            results_by_file[str(image_file)] = result
            
        except Exception as e:
            error_msg = str(e)
            failed_files.append((str(image_file), error_msg))
            
            if not silent:
                print(f"\nError processing {image_file.name}: {error_msg}")
            
            if not continue_on_error:
                raise
    
    # Print summary
    if not silent:
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

