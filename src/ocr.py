import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm

# Cache for models
_cached_processor = None
_cached_model = None
_cached_key = None

def _validate_device(device):
    """Validate device availability."""
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device

def _get_cache_key(model_id, device):
    """Create cache key tuple."""
    return (model_id, device)

def load_ocr_model(model_id="jzhang533/PaddleOCR-VL-For-Manga", device=None, silent=False):
    """
    Load the PaddleOCR-VL model for text extraction

    Args:
        model_id: Hugging Face model ID
        device: Device to use ("cuda" or "cpu"), auto-detects if None
        silent: If True, suppress progress messages

    Returns:
        processor, model: Loaded processor and model
    """
    global _cached_processor, _cached_model, _cached_key

    device = _validate_device(device)
    cache_key = _get_cache_key(model_id, device)

    # Return cached models if same model and device
    if _cached_model is not None and cache_key == _cached_key:
        return _cached_processor, _cached_model

    if not silent:
        print(f"Loading OCR model: {model_id} on {device}")
        print(f"OCR model will be cached in HuggingFace cache directory")

    _cached_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)

    _cached_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    _cached_key = cache_key

    if not silent:
        print(f"OCR model loaded successfully on {device}")

    return _cached_processor, _cached_model

def extract_text(image, model_id="jzhang533/PaddleOCR-VL-For-Manga", max_new_tokens=2048, device=None):
    """
    Extract text from an image using PaddleOCR-VL

    Args:
        image: PIL Image object
        model_id: Hugging Face model ID
        max_new_tokens: Maximum tokens to generate
        device: Device to use ("cuda" or "cpu"), auto-detects if None

    Returns:
        extracted_text: Extracted text string
    """
    processor, model = load_ocr_model(model_id, device=device)
    
    # Prepare the input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "OCR:"},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {
        k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }
    
    # Run inference
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    
    # Decode result
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = generated_ids[:, input_length:]
    extracted_text = processor.batch_decode(
        generated_tokens, skip_special_tokens=True
    )[0]
    
    return extracted_text


def extract_text_batch(images, model_id="jzhang533/PaddleOCR-VL-For-Manga", max_new_tokens=2048, device=None, silent=False, progress_callback=None):
    """
    Extract text from multiple images using PaddleOCR-VL (sequential processing for debugging)

    Processes each image individually by calling extract_text on each one sequentially.
    This is useful for debugging batch processing issues.

    Args:
        images: List of PIL Image objects
        model_id: Hugging Face model ID
        max_new_tokens: Maximum tokens to generate
        device: Device to use ("cuda" or "cpu"), auto-detects if None
        silent: If True, suppress progress messages
        progress_callback: Optional callback function to call after each image (receives no arguments)

    Returns:
        List of extracted text strings (one per image)
    """
    if not images:
        return []

    # Pre-load the model once before processing all images
    load_ocr_model(model_id, device=device, silent=silent)

    extracted_texts = []

    # Use progress_callback if provided, otherwise use tqdm
    if progress_callback:
        iterator = images
    else:
        iterator = tqdm(images, desc="Extracting text", disable=silent, unit="bubble")

    for idx, image in enumerate(iterator):
        try:
            extracted_text = extract_text(image, model_id=model_id, max_new_tokens=max_new_tokens, device=device)
            extracted_texts.append(extracted_text)
        except Exception as e:
            if not silent:
                print(f"Warning: Error extracting text from bubble {idx+1}: {e}")
            extracted_texts.append("")

        # Call progress callback if provided
        if progress_callback:
            progress_callback()

    return extracted_texts


# def extract_text_batch(images, model_id="jzhang533/PaddleOCR-VL-For-Manga", max_new_tokens=2048, silent=False):
#     """
#     Extract text from multiple images using PaddleOCR-VL (batch processing)
    
#     Args:
#         images: List of PIL Image objects
#         model_id: Hugging Face model ID
#         max_new_tokens: Maximum tokens to generate
#         silent: If True, suppress progress messages
    
#     Returns:
#         List of extracted text strings (one per image)
#     """
#     if not images:
#         return []
    
#     processor, model = load_ocr_model(model_id)
    
#     # Max batch size constant
#     MAX_BATCH_SIZE = 2
    
#     def run_batch_ocr(images, model, processor, max_new_tokens=512):
#         """
#         Args:
#             images: List of PIL.Image objects
#         """
        
#         # 1. Prepare batch messages
#         # We create a separate conversation list for every image in the batch
#         batch_messages = []
#         for image in images:
#             batch_messages.append([
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "image", "image": image},
#                         {"type": "text", "text": "OCR:"},
#                     ],
#                 }
#             ])

#         # 2. Apply chat template to creating prompt strings
#         texts = [
#             processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
#             for msg in batch_messages
#         ]

#         # 3. Process inputs with Padding
#         # 'padding=True' is essential here to handle different image aspect ratios or prompt lengths
#         inputs = processor(
#             text=texts, 
#             images=images, 
#             return_tensors="pt", 
#             padding=True
#         )

#         # Move to device
#         inputs = {k: v.to(model.device) for k, v in inputs.items()}

#         # 4. Generate
#         with torch.inference_mode():
#             generated_ids = model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=False,
#                 use_cache=True,
#             )

#         # 5. Decode Batch
#         # Slice only the new tokens. 
#         # Note: We rely on input_ids length, but be careful with padding tokens in the input.
#         input_length = inputs["input_ids"].shape[1]
#         generated_tokens = generated_ids[:, input_length:]
        
#         extracted_texts = processor.batch_decode(
#             generated_tokens, skip_special_tokens=True
#         )

#         return extracted_texts
    
#     # Process in batches if needed
#     all_extracted_texts = []
#     for i in range(0, len(images), MAX_BATCH_SIZE):
#         batch = images[i:i + MAX_BATCH_SIZE]
#         if not silent:
#             print(f"Extracting text from batch {i // MAX_BATCH_SIZE + 1} ({len(batch)} image(s))...")
#         batch_results = run_batch_ocr(batch, model, processor, max_new_tokens=max_new_tokens)
#         all_extracted_texts.extend(batch_results)
    
#     return all_extracted_texts
