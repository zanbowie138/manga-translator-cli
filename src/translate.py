import ctranslate2
import sentencepiece
from huggingface_hub import snapshot_download
import os
import re

# Cache for loaded models
_translator = None
_tokenizer_source = None
_tokenizer_target = None
_model_path = None
_device = None

# Japanese character range pattern (excluding middle dot ・ U+30FB)
# https://stackoverflow.com/questions/30069846/how-to-find-out-chinese-or-japanese-character-in-a-string-in-python
CJK_CHAR_PATTERN = re.compile(r"[\u3300-\u33ff\ufe30-\ufe4f\uf900-\ufaff\U0002f800-\U0002fa1f\u30a0-\u30fa\u30fc-\u30ff\u2e80-\u2eff\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf]+")

def _load_models(model_path=None, device='cpu'):
    """Load translation models lazily (only once)"""
    global _translator, _tokenizer_source, _tokenizer_target, _model_path, _device
    
    if model_path is None:
        model_path = 'sugoi-v4-ja-en-ctranslate2'
    
    # Download model if not exists
    if not os.path.exists(model_path):
        print(f"Downloading translation model to {model_path}...")
        snapshot_download(repo_id='entai2965/sugoi-v4-ja-en-ctranslate2', local_dir=model_path)
    
    # Return cached models if already loaded
    if _translator is not None and _model_path == model_path and _device == device:
        return _translator, _tokenizer_source, _tokenizer_target
    
    # Load models
    print(f"Loading translation models from {model_path}...")
    sentencepiece_model_path = os.path.join(model_path, 'spm')
    
    _translator = ctranslate2.Translator(model_path, device=device)
    _tokenizer_source = sentencepiece.SentencePieceProcessor(
        os.path.join(sentencepiece_model_path, 'spm.ja.nopretok.model')
    )
    _tokenizer_target = sentencepiece.SentencePieceProcessor(
        os.path.join(sentencepiece_model_path, 'spm.en.nopretok.model')
    )
    
    _model_path = model_path
    _device = device
    
    return _translator, _tokenizer_source, _tokenizer_target

def translate_phrase(text, model_path=None, device='cpu', beam_size=5):
    """
    Translate a Japanese phrase to English
    
    Args:
        text: Japanese text string to translate
        model_path: Path to model directory (defaults to 'sugoi-v4-ja-en-ctranslate2')
        device: Device to use ('cpu' or 'cuda')
        beam_size: Beam size for translation (default 5)
    
    Returns:
        Tuple of (translated_text, success) where:
        - translated_text: Translated English text string (empty if no CJK found)
        - success: True if text contains CJK characters and translation was successful, False otherwise
    """
    # Check if text contains CJK characters
    if not text or not text.strip():
        return "", False
    
    has_cjk = bool(CJK_CHAR_PATTERN.search(text))
    if not has_cjk:
        return "", False
    
    translator, tokenizer_source, tokenizer_target = _load_models(model_path, device)
    
    # Tokenize
    tokenized = tokenizer_source.encode(text, out_type=str)
    
    # Translate
    translated = translator.translate_batch(source=[tokenized], beam_size=beam_size)
    
    # Decode
    translated_text = tokenizer_target.decode(translated[0].hypotheses[0]).replace('<unk>', '')
    
    return translated_text, True
