import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

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
    
    print(f"Loading model: {model_id} on {device}...")
    
    _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
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
    print("Extracting text...")
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
    Extract text from multiple images using PaddleOCR-VL (batch processing)
    
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
    
    processor, model = load_ocr_model(model_id)
    
    # Prepare batch inputs
    texts = []
    image_list = []
    for image in images:
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
        texts.append(text)
        image_list.append(image)
    
    # Process batch
    inputs = processor(text=texts, images=image_list, return_tensors="pt", padding=True)
    inputs = {
        k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }
    
    # Run batch inference
    if not silent:
        print(f"Extracting text from {len(images)} image(s)...")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    
    # Decode results
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = generated_ids[:, input_length:]
    extracted_texts = processor.batch_decode(
        generated_tokens, skip_special_tokens=True
    )
    
    return extracted_texts
