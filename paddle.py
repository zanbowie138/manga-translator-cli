import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from paddleocr import PaddleOCR, LayoutDetection
import numpy as np
import sys

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

def extract_text(image_path, model_id="jzhang533/PaddleOCR-VL-For-Manga", max_new_tokens=2048):
    """
    Extract text from an image using PaddleOCR-VL
    
    Args:
        image_path: Path to input image
        model_id: Hugging Face model ID
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        extracted_text: Extracted text string
    """
    processor, model = load_ocr_model(model_id)
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
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

def detect_layout(image_path, model_name="PP-DocLayout-L", output_dir="output"):
    """
    Detect document layout using PaddleOCR LayoutDetection
    
    Args:
        image_path: Path to input image
        model_name: Layout detection model name
        output_dir: Directory to save results
    
    Returns:
        output: Layout detection results
    """
    model = LayoutDetection(model_name=model_name)
    output = model.predict(input=image_path, batch_size=1)
    
    # Save results
    for res in output:
        res.print()
        res.save_to_img(save_path=output_dir)
        res.save_to_json(save_path=f"{output_dir}/res.json")
    
    return output

def detect_text_regions(image_path):
    """
    Detect text regions using PaddleOCR (basic detection)
    
    Args:
        image_path: Path to input image
    
    Returns:
        detections: List of text region detections
        texts: List of recognized texts
    """
    detector = PaddleOCR()
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)
    
    detection_result = detector.predict(image_array)
    
    # Extract OCRResult object
    if isinstance(detection_result, list) and len(detection_result) > 0:
        ocr_result = detection_result[0]
        detections = ocr_result.get('dt_polys', ocr_result.get('rec_polys', []))
        texts = ocr_result.get('rec_texts', [])
        return detections, texts
    else:
        return [], []

