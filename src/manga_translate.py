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

from src.bubble_detect import load_bubble_detection_model, run_detection
from src.image_utils import get_cropped_images, save_pil_image
from src.bbox import BoundingBox, remove_parent_boxes, combine_overlapping_bubbles
from src.draw_text import draw_text_on_image
from src.bubble_clean import fill_bubble_interiors, get_bubble_text_mask
from src.config import Config
from src.console import Console
from src.output_manager import OutputManager


def _setup_output_structure(
    output_folder: str,
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
        output_subfolder: Optional subfolder name
        save_speech_bubbles: Whether to create speech_bubbles folder
        save_bubble_interiors: Whether to create bubble_masks folder
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
        _get_output_dir_path(output_base, "bubble_masks", output_subfolder).mkdir(parents=True, exist_ok=True)
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
    Create visualization of bubble interiors by painting bubble masks with transparent blue and green.
    
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
        print("Creating bubble masks visualization...")
    
    # Use visualize_bubble_masks with pre-generated masks
    from src.bubble_clean import visualize_bubble_masks
    
    # Extract boxes from bubble_texts
    boxes = [bbox for bbox, _ in bubble_texts]
    
    # visualize_bubble_masks returns BGRA numpy array
    output_bgra = visualize_bubble_masks(
        image=input_image,
        boxes=boxes,
        bubble_masks=bubble_masks
    )
    
    # Convert BGRA numpy array to RGBA PIL Image
    bubble_masks_rgba = cv2.cvtColor(output_bgra, cv2.COLOR_BGRA2RGBA)
    bubble_masks_pil = Image.fromarray(bubble_masks_rgba)
    
    output_paths = {}
    bubble_masks_path = _get_output_file_path(
        output_base, "bubble_masks", f"{output_filename}_bubble_masks.png", output_subfolder
    )
    save_pil_image(bubble_masks_pil, str(bubble_masks_path), print_message=not silent)
    output_paths['bubble_masks'] = str(bubble_masks_path)
    
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
    
    # Step 1: Batch extract text from all bubbles
    from src.ocr import extract_text_batch
    
    # Prepare images and track indices
    images_to_process = []
    bbox_indices = []  # List of (bbox, index) tuples corresponding to images
    
    for i, (cropped_img, bbox) in enumerate(cropped_images, 1):
        images_to_process.append(cropped_img)
        bbox_indices.append((bbox, i))
        if not silent:
            print(f"Preparing bubble {i} (bbox: {bbox})")
    
    # Batch extract all texts
    extracted_texts = []  # List of (bbox, extracted_text, index) tuples
    try:
        extracted_text_list = extract_text_batch(
            images_to_process,
            model_id=ocr_model_id,
            max_new_tokens=ocr_max_new_tokens,
            device=translation_device,
            silent=silent
        )
        
        # Match extracted texts back to bubbles
        for (bbox, index), extracted_text in zip(bbox_indices, extracted_text_list):
            if not silent:
                sys.stdout.reconfigure(encoding='utf-8')
                print(f"\nBubble {index} (bbox: {bbox}):")
                print(f"  Extracted text: {extracted_text}")
            extracted_texts.append((bbox, extracted_text, index))
    except Exception as e:
        if not silent:
            print(f"Error in batch OCR extraction: {e}")
        # Fallback to empty texts
        for bbox, index in bbox_indices:
            extracted_texts.append((bbox, "", index))
    
    # Step 2: Extract texts for translation
    texts_list = [extracted_text for _, extracted_text, _ in extracted_texts]
    
    # Step 3: Translate all texts individually
    from src.translate import translate_batch
    
    if not silent:
        print("\n" + "=" * 50)
        print("Translating texts...")
        print("=" * 50)
    
    translated_texts_list = translate_batch(
        texts_list,
        model_path=translation_model_path,
        device=translation_device,
        beam_size=translation_beam_size,
        silent=silent
    )
    
    # Step 4: Map translations back to bubbles
    bubble_texts = []  # List of (bbox, translated_text) tuples
    japanese_bboxes = []  # Track which bubbles contain Japanese
    
    for (bbox, extracted_text, index), translated_text in zip(extracted_texts, translated_texts_list):
        if translated_text.strip():
            bubble_texts.append((bbox, translated_text))
            japanese_bboxes.append(bbox)
            if not silent:
                print(f"\nBubble {index}:")
                print(f"  Original: {extracted_text}")
                print(f"  Translated: {translated_text}")
    
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
    config: Config,
    console: Console,
    detection_model=None,  # Optional: pass pre-loaded model or None to load
    output_filename: Optional[str] = None,  # If None, use input filename
    output_subfolder: Optional[str] = None  # Optional subfolder name (e.g., for folder processing)
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
        config: Configuration object containing pipeline parameters
        console: Console object for output messages
        detection_model: Pre-loaded YOLO model (None to load automatically)
        output_filename: Custom output filename (None to derive from input)
        output_subfolder: Optional subfolder name for organizing outputs (e.g., input folder name)

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
    output_manager = OutputManager(output_folder, output_subfolder)
    output_manager.setup(
        save_speech_bubbles=config.save_speech_bubbles,
        save_bubble_interiors=config.save_bubble_interiors,
        save_cleaned=config.save_cleaned,
        save_translated=config.save_translated
    )
    output_base = output_manager.base

    try:
        # Load the input image
        input_image = Image.open(input_image_path).convert("RGB")

        # Load detection model
        if detection_model is None:
            console.info("Loading detection model...")
            detection_model = load_bubble_detection_model(silent=console.quiet)

        # Detect speech bubbles
        annotated_bubble_img, boxes, output_paths = _detect_speech_bubbles(
            input_image,
            detection_model,
            config.conf_threshold,
            config.iou_threshold,
            config.save_speech_bubbles,
            output_base,
            output_filename,
            output_subfolder,
            console.quiet
        )

        results['annotated_bubble_image'] = annotated_bubble_img
        results['boxes'] = boxes
        results['output_paths'].update(output_paths)

        # Process bounding boxes based on mode
        filtered_bboxes = _process_bounding_boxes(
            boxes,
            config.bbox_processing,
            config.parent_box_threshold,
            console.quiet
        )

        results['filtered_boxes'] = filtered_bboxes

        if not filtered_bboxes:
            if config.save_translated:
                translated_path = _get_output_file_path(
                    output_base, "translated", f"{output_filename}_translated.png", output_subfolder
                )
                save_pil_image(input_image, str(translated_path), print_message=not console.quiet)
                results['output_paths']['translated'] = str(translated_path)
                results['translated_image'] = input_image
            return results

        # Crop individual speech bubbles
        console.info("Cropping speech bubbles...")
        cropped_images = get_cropped_images(input_image, filtered_bboxes)
        console.info(f"Cropped {len(cropped_images)} speech bubble images")

        # Extract and translate text
        bubble_texts, japanese_bboxes = _extract_and_translate_text(
            cropped_images,
            config.ocr_model_id,
            config.ocr_max_new_tokens,
            config.translation_model_path,
            config.translation_device,
            config.translation_beam_size,
            console.quiet
        )

        results['bubble_texts'] = bubble_texts

        # Check if we have any Japanese text to translate
        if not bubble_texts:
            console.info("No Japanese text found in any bubbles.")
            return results

        # Generate bubble masks for drawing
        bubble_masks = _generate_bubble_masks(
            input_image,
            bubble_texts,
            config.threshold_value,
            console.quiet
        )

        results['bubble_masks'] = bubble_masks

        # Create bubble interiors visualization using pre-generated masks
        bubble_interiors_pil, output_paths = _create_bubble_interiors_visualization(
            input_image,
            bubble_texts,
            bubble_masks,
            config.save_bubble_interiors,
            output_base,
            output_filename,
            output_subfolder,
            console.quiet
        )

        if bubble_interiors_pil is not None:
            results['bubble_interiors_visualization'] = bubble_interiors_pil
        results['output_paths'].update(output_paths)

        # Clean bubble interiors
        cleaned_image_pil, output_paths = _clean_bubble_interiors(
            input_image,
            japanese_bboxes,
            config.threshold_value,
            config.save_cleaned,
            output_base,
            output_filename,
            output_subfolder,
            console.quiet
        )

        results['cleaned_image'] = cleaned_image_pil
        results['output_paths'].update(output_paths)

        # Draw translated text
        translated_image, output_paths = _draw_translated_text(
            cleaned_image_pil,
            bubble_texts,
            bubble_masks,
            config.font_path,
            config.save_translated,
            output_base,
            output_filename,
            output_subfolder,
            console.quiet
        )

        results['translated_image'] = translated_image
        results['output_paths'].update(output_paths)

        console.section("Processing complete!")

    except Exception as e:
        console.error(f"Error in manga translation: {e}")
        raise

    return results


def translate_manga_page_batch(
    input_image_paths: List[str],
    output_folder: str,
    config: Config,
    console: Console,
    detection_model=None,  # Optional: pass pre-loaded model or None to load
    output_subfolder: Optional[str] = None,  # Optional subfolder name (e.g., for folder processing)
    batch_amount: Optional[int] = None  # Process in chunks if specified
) -> Dict[str, Dict[str, Any]]:
    """
    Batch process multiple manga pages through each pipeline step together.

    This function processes all pages simultaneously through each step:
    1. Loads all images
    2. Detects bubbles for all images
    3. Processes bounding boxes for all images
    4. Extracts text from all bubbles (batch OCR)
    5. Translates all texts (batch translation)
    6. Generates masks for all bubbles
    7. Cleans all images
    8. Draws text on all images

    Args:
        input_image_paths: List of paths to input manga page images
        output_folder: Base folder path for all outputs
        config: Configuration object containing pipeline parameters
        console: Console object for output messages
        detection_model: Pre-loaded YOLO model (None to load automatically)
        output_subfolder: Optional subfolder name for organizing outputs (e.g., input folder name)
        batch_amount: Maximum number of pages to process in each batch (None = process all at once)

    Returns:
        Dictionary mapping each input path to its result dictionary (same format as translate_manga_page)
    """
    if not input_image_paths:
        return {}

    # Handle batch_amount chunking
    if batch_amount is not None and batch_amount > 0:
        all_results = {}
        for i in range(0, len(input_image_paths), batch_amount):
            chunk = input_image_paths[i:i + batch_amount]
            console.section(f"Processing batch {i//batch_amount + 1} ({len(chunk)} pages)...")
            chunk_results = translate_manga_page_batch(
                chunk,
                output_folder,
                config=config,
                console=console,
                detection_model=detection_model,
                output_subfolder=output_subfolder,
                batch_amount=None  # Don't recurse
            )
            all_results.update(chunk_results)
        return all_results

    # Initialize results dictionary
    results_by_path: Dict[str, Dict[str, Any]] = {}

    # Set up output folder structure
    output_manager = OutputManager(output_folder, output_subfolder)
    output_manager.setup(
        save_speech_bubbles=config.save_speech_bubbles,
        save_bubble_interiors=config.save_bubble_interiors,
        save_cleaned=config.save_cleaned,
        save_translated=config.save_translated
    )
    output_base = output_manager.base
    
    try:
        # Step 1: Load all images
        console.section(f"Loading {len(input_image_paths)} image(s)...")

        input_images = []
        output_filenames = []
        for input_path_str in input_image_paths:
            input_path = Path(input_path_str)
            input_image = Image.open(input_path_str).convert("RGB")
            input_images.append(input_image)
            output_filenames.append(input_path.stem)

            # Initialize result dictionary for this page
            results_by_path[input_path_str] = {
                'annotated_bubble_image': None,
                'boxes': [],
                'filtered_boxes': [],
                'blue_image': None,
                'cleaned_image': None,
                'translated_image': None,
                'bubble_texts': [],
                'bubble_masks': {},
                'bubble_interiors_visualization': None,
                'output_paths': {}
            }

        # Step 2: Load detection model
        if detection_model is None:
            console.info("Loading detection model...")
            detection_model = load_bubble_detection_model(silent=console.quiet)

        # Step 3: Detect bubbles for all images
        console.section(f"Detecting speech bubbles for {len(input_images)} image(s)...")

        all_annotated_images = []
        all_boxes = []
        all_output_paths = []

        for page_idx, (input_image, input_path_str, output_filename) in enumerate(zip(input_images, input_image_paths, output_filenames), 1):
            console.info(f"Page {page_idx}/{len(input_images)}: {Path(input_path_str).name}")

            annotated_bubble_img, boxes, output_paths = _detect_speech_bubbles(
                input_image,
                detection_model,
                config.conf_threshold,
                config.iou_threshold,
                config.save_speech_bubbles,
                output_base,
                output_filename,
                output_subfolder,
                console.quiet
            )

            all_annotated_images.append(annotated_bubble_img)
            all_boxes.append(boxes)
            all_output_paths.append(output_paths)

            results_by_path[input_path_str]['annotated_bubble_image'] = annotated_bubble_img
            results_by_path[input_path_str]['boxes'] = boxes
            results_by_path[input_path_str]['output_paths'].update(output_paths)

        # Step 4: Process bounding boxes for all images
        console.section("Processing bounding boxes for all images...")

        all_filtered_bboxes = []
        for page_idx, (boxes, input_path_str) in enumerate(zip(all_boxes, input_image_paths), 1):
            filtered_bboxes = _process_bounding_boxes(
                boxes,
                config.bbox_processing,
                config.parent_box_threshold,
                console.quiet
            )
            all_filtered_bboxes.append(filtered_bboxes)
            results_by_path[input_path_str]['filtered_boxes'] = filtered_bboxes

        # Step 5: Crop all bubbles from all images and track page indices
        console.section("Cropping speech bubbles from all images...")

        all_cropped_images = []  # List of (page_idx, cropped_img, bbox) tuples
        page_bubble_indices = []  # List of (start_idx, end_idx) for each page

        for page_idx, (input_image, filtered_bboxes) in enumerate(zip(input_images, all_filtered_bboxes)):
            if not filtered_bboxes:
                page_bubble_indices.append((len(all_cropped_images), len(all_cropped_images)))
                continue

            start_idx = len(all_cropped_images)
            cropped_images = get_cropped_images(input_image, filtered_bboxes)
            for cropped_img, bbox in cropped_images:
                all_cropped_images.append((page_idx, cropped_img, bbox))
            end_idx = len(all_cropped_images)
            page_bubble_indices.append((start_idx, end_idx))

            console.info(f"Page {page_idx + 1}: Cropped {len(cropped_images)} speech bubbles")

        if not all_cropped_images:
            console.info("No speech bubbles found in any images.")
            if config.save_translated:
                for input_image, input_path_str, output_filename in zip(
                    input_images, input_image_paths, output_filenames
                ):
                    translated_path = _get_output_file_path(
                        output_base, "translated", f"{output_filename}_translated.png", output_subfolder
                    )
                    save_pil_image(input_image, str(translated_path), print_message=not console.quiet)
                    results_by_path[input_path_str]['output_paths']['translated'] = str(translated_path)
                    results_by_path[input_path_str]['translated_image'] = input_image
            return results_by_path

        # Step 6: Batch OCR all bubbles across all images
        console.section(f"Running OCR on {len(all_cropped_images)} speech bubbles (batch processing)...")

        from src.ocr import extract_text_batch

        # Prepare images for batch OCR
        ocr_images = [cropped_img for _, cropped_img, _ in all_cropped_images]
        ocr_results = extract_text_batch(
            ocr_images,
            model_id=config.ocr_model_id,
            max_new_tokens=config.ocr_max_new_tokens,
            device=config.translation_device,
            silent=console.quiet
        )

        # Step 7: Extract texts for batch translation
        all_extracted_texts = []  # List of (page_idx, bbox, extracted_text, bubble_index) tuples
        bubble_counter = {}  # Track bubble index per page

        for (page_idx, _, bbox), extracted_text in zip(all_cropped_images, ocr_results):
            # Track bubble index per page (starting from 1)
            if page_idx not in bubble_counter:
                bubble_counter[page_idx] = 0
            bubble_counter[page_idx] += 1
            bubble_index = bubble_counter[page_idx]

            all_extracted_texts.append((page_idx, bbox, extracted_text, bubble_index))
            if not console.quiet:
                sys.stdout.reconfigure(encoding='utf-8')
                console.print(f"\nPage {page_idx + 1}, Bubble {bubble_index} (bbox: {bbox}):")
                console.print(f"  Extracted text: {extracted_text}")

        # Extract texts for translation
        texts_list = [extracted_text for _, _, extracted_text, _ in all_extracted_texts]

        # Translate all texts individually
        from src.translate import translate_batch

        console.section("Translating texts...")

        translated_texts_list = translate_batch(
            texts_list,
            model_path=config.translation_model_path,
            device=config.translation_device,
            beam_size=config.translation_beam_size,
            silent=console.quiet
        )

        # Map translations back to pages and bubbles
        all_bubble_texts = [[] for _ in input_images]  # List of lists: one per page
        all_japanese_bboxes = [[] for _ in input_images]  # List of lists: one per page

        for (page_idx, bbox, extracted_text, bubble_index), translated_text in zip(all_extracted_texts, translated_texts_list):
            if translated_text.strip():
                all_bubble_texts[page_idx].append((bbox, translated_text))
                all_japanese_bboxes[page_idx].append(bbox)
                if not console.quiet:
                    console.print(f"\nPage {page_idx + 1}, Bubble {bubble_index}:")
                    console.print(f"  Original: {extracted_text}")
                    console.print(f"  Translated: {translated_text}")

        # Step 8: Generate masks for all bubbles across all images
        console.section("Generating bubble masks for all images...")

        all_bubble_masks = []
        for page_idx, (input_image, bubble_texts) in enumerate(zip(input_images, all_bubble_texts)):
            if bubble_texts:
                bubble_masks = _generate_bubble_masks(
                    input_image,
                    bubble_texts,
                    config.threshold_value,
                    console.quiet
                )
                all_bubble_masks.append(bubble_masks)
                results_by_path[input_image_paths[page_idx]]['bubble_masks'] = bubble_masks
            else:
                all_bubble_masks.append({})

        # Step 9: Create bubble interiors visualizations
        for page_idx, (input_image, bubble_texts, bubble_masks, input_path_str, output_filename) in enumerate(
            zip(input_images, all_bubble_texts, all_bubble_masks, input_image_paths, output_filenames)
        ):
            if bubble_texts and bubble_masks:
                bubble_interiors_pil, output_paths = _create_bubble_interiors_visualization(
                    input_image,
                    bubble_texts,
                    bubble_masks,
                    config.save_bubble_interiors,
                    output_base,
                    output_filename,
                    output_subfolder,
                    console.quiet
                )
                if bubble_interiors_pil is not None:
                    results_by_path[input_path_str]['bubble_interiors_visualization'] = bubble_interiors_pil
                results_by_path[input_path_str]['output_paths'].update(output_paths)

        # Step 10: Clean all images
        console.section("Clearing speech bubbles for all images...")

        all_cleaned_images = []
        for page_idx, (input_image, japanese_bboxes, input_path_str, output_filename) in enumerate(
            zip(input_images, all_japanese_bboxes, input_image_paths, output_filenames)
        ):
            if japanese_bboxes:
                cleaned_image_pil, output_paths = _clean_bubble_interiors(
                    input_image,
                    japanese_bboxes,
                    config.threshold_value,
                    config.save_cleaned,
                    output_base,
                    output_filename,
                    output_subfolder,
                    console.quiet
                )
                all_cleaned_images.append(cleaned_image_pil)
                results_by_path[input_path_str]['cleaned_image'] = cleaned_image_pil
                results_by_path[input_path_str]['output_paths'].update(output_paths)
            else:
                all_cleaned_images.append(input_image)

        # Step 11: Draw translated text on all images
        console.section("Drawing translated text on all images...")

        for page_idx, (cleaned_image, bubble_texts, bubble_masks, input_path_str, output_filename) in enumerate(
            zip(all_cleaned_images, all_bubble_texts, all_bubble_masks, input_image_paths, output_filenames)
        ):
            if bubble_texts and bubble_masks:
                translated_image, output_paths = _draw_translated_text(
                    cleaned_image,
                    bubble_texts,
                    bubble_masks,
                    config.font_path,
                    config.save_translated,
                    output_base,
                    output_filename,
                    output_subfolder,
                    console.quiet
                )
                results_by_path[input_path_str]['translated_image'] = translated_image
                results_by_path[input_path_str]['bubble_texts'] = bubble_texts
                results_by_path[input_path_str]['output_paths'].update(output_paths)
            elif config.save_translated:
                translated_path = _get_output_file_path(
                    output_base, "translated", f"{output_filename}_translated.png", output_subfolder
                )
                save_pil_image(cleaned_image, str(translated_path), print_message=not console.quiet)
                results_by_path[input_path_str]['output_paths']['translated'] = str(translated_path)
                results_by_path[input_path_str]['translated_image'] = cleaned_image

        console.section(f"Batch processing complete! Processed {len(input_image_paths)} page(s)")

    except Exception as e:
        console.error(f"Error in batch manga translation: {e}")
        raise

    return results_by_path


def translate_manga_folder(
    input_folder: str,
    output_folder: str,
    config: Config,
    console: Console
) -> Dict[str, Any]:
    """
    Translate all manga pages in a folder from Japanese to English.

    Processes all supported image files in the input folder, reusing the detection
    model across pages for efficiency. Each page is processed using translate_manga_page.

    Args:
        input_folder: Path to folder containing manga page images
        output_folder: Base folder path for all outputs (same structure as translate_manga_page)
        config: Configuration object containing pipeline parameters
        console: Console object for output messages

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
        console.info(f"No image files found in {input_folder}")
        return {
            'processed_files': [],
            'failed_files': [],
            'results_by_file': {},
            'total_files': 0,
            'successful_count': 0,
            'failed_count': 0
        }

    console.section(f"Found {total_files} image file(s) to process")

    # Load detection model once for reuse across all pages
    detection_model = None
    console.info("Loading detection model (will be reused for all pages)...")
    try:
        detection_model = load_bubble_detection_model(silent=console.quiet)
    except Exception as e:
        if config.stop_on_error:
            raise
        console.error(f"Warning: Could not load detection model: {e}")

    # Process each image file
    processed_files = []
    failed_files = []
    results_by_file = {}

    for i, image_file in enumerate(image_files, 1):
        console.section(f"Processing file {i}/{total_files}: {image_file.name}")

        try:
            # Use translate_manga_page with pre-loaded model
            result = translate_manga_page(
                input_image_path=str(image_file),
                output_folder=output_folder,
                config=config,
                console=console,
                detection_model=detection_model,
                output_filename=None,  # Use input filename
                output_subfolder=folder_name  # Organize outputs by folder name
            )

            processed_files.append(str(image_file))
            results_by_file[str(image_file)] = result

        except Exception as e:
            error_msg = str(e)
            failed_files.append((str(image_file), error_msg))

            console.error(f"Error processing {image_file.name}: {error_msg}")

            if config.stop_on_error:
                raise

    # Print summary
    console.section("Folder processing complete!")
    console.info(f"Total files: {total_files}")
    console.success(f"Successfully processed: {len(processed_files)}")
    if failed_files:
        console.error(f"Failed: {len(failed_files)}")
        console.print("\nFailed files:")
        for file_path, error in failed_files:
            console.print(f"  - {Path(file_path).name}: {error}")

    return {
        'processed_files': processed_files,
        'failed_files': failed_files,
        'results_by_file': results_by_file,
        'total_files': total_files,
        'successful_count': len(processed_files),
        'failed_count': len(failed_files)
    }


def translate_manga_folder_batch(
    input_folder: str,
    output_folder: str,
    config: Config,
    console: Console,
    batch_amount: Optional[int] = None
) -> Dict[str, Any]:
    """
    Translate all manga pages in a folder using batch processing.

    Processes all supported image files in the input folder using batch processing,
    where all pages go through each pipeline step together for efficiency.

    Args:
        input_folder: Path to folder containing manga page images
        output_folder: Base folder path for all outputs
        config: Configuration object containing pipeline parameters
        console: Console object for output messages
        batch_amount: Maximum number of pages to process in each batch (None = process all at once)

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
        console.info(f"No image files found in {input_folder}")
        return {
            'processed_files': [],
            'failed_files': [],
            'results_by_file': {},
            'total_files': 0,
            'successful_count': 0,
            'failed_count': 0
        }

    console.section(f"Found {total_files} image file(s) to process (batch mode)")

    # Load detection model once for reuse across all pages
    detection_model = None
    console.info("Loading detection model (will be reused for all pages)...")
    try:
        detection_model = load_bubble_detection_model(silent=console.quiet)
    except Exception as e:
        console.error(f"Warning: Could not load detection model: {e}")
        raise

    # Process all files using batch processing
    processed_files = []
    failed_files = []
    results_by_file = {}

    # Convert to list of string paths
    image_paths = [str(f) for f in image_files]

    try:
        # Call batch processing function
        batch_results = translate_manga_page_batch(
            input_image_paths=image_paths,
            output_folder=output_folder,
            config=config,
            console=console,
            detection_model=detection_model,
            output_subfolder=folder_name,
            batch_amount=batch_amount
        )

        # Process results
        for file_path, result in batch_results.items():
            processed_files.append(file_path)
            results_by_file[file_path] = result

    except Exception as e:
        error_msg = str(e)
        # Mark all remaining files as failed
        for file_path in image_paths:
            if file_path not in processed_files:
                failed_files.append((file_path, error_msg))
        console.error(f"Error in batch processing: {error_msg}")
        raise

    # Print summary
    console.section("Batch folder processing complete!")
    console.info(f"Total files: {total_files}")
    console.success(f"Successfully processed: {len(processed_files)}")
    if failed_files:
        console.error(f"Failed: {len(failed_files)}")
        console.print("\nFailed files:")
        for file_path, error in failed_files:
            console.print(f"  - {Path(file_path).name}: {error}")

    return {
        'processed_files': processed_files,
        'failed_files': failed_files,
        'results_by_file': results_by_file,
        'total_files': total_files,
        'successful_count': len(processed_files),
        'failed_count': len(failed_files)
    }

