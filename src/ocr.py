import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm

# Cache for models
_processor = None
_model = None
_device = None
_model_id = None

def load_ocr_model(model_id="jzhang533/PaddleOCR-VL-For-Manga", device=None):
    """
    Load the PaddleOCR-VL model for text extraction
    
    Args:
        model_id: Hugging Face model ID
        device: Device to use ("cuda" or "cpu"), auto-detects if None
    
    Returns:
        processor, model: Loaded processor and model
    """
    global _processor, _model, _device, _model_id
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Return cached models if same model and device
    if _model is not None and _model_id == model_id and _device == device:
        return _processor, _model

    _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    _device = device
    _model_id = model_id
    
    return _processor, _model

def extract_text(image, model_id="jzhang533/PaddleOCR-VL-For-Manga", max_new_tokens=2048):
    """
    Extract text from an image using PaddleOCR-VL
    
    Args:
        image: PIL Image object
        model_id: Hugging Face model ID
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        extracted_text: Extracted text string
    """
    processor, model = load_ocr_model(model_id)
    
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


def extract_text_batch(images, model_id="jzhang533/PaddleOCR-VL-For-Manga", max_new_tokens=2048, silent=False):
    """
    Extract text from multiple images using PaddleOCR-VL (sequential processing for debugging)

    Processes each image individually by calling extract_text on each one sequentially.
    This is useful for debugging batch processing issues.

    Args:
        images: List of PIL Image objects
        model_id: Hugging Face model ID
        max_new_tokens: Maximum tokens to generate
        silent: If True, suppress progress messages

    Returns:
        List of extracted text strings (one per image)
    """
    if not images:
        return []

    extracted_texts = []
    iterator = tqdm(images, desc="Extracting text", disable=silent, unit="bubble")

    for image in iterator:
        try:
            extracted_text = extract_text(image, model_id=model_id, max_new_tokens=max_new_tokens)
            extracted_texts.append(extracted_text)
        except Exception as e:
            extracted_texts.append("")

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
