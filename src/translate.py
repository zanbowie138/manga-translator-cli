import ctranslate2
import sentencepiece
from huggingface_hub import snapshot_download
import os

# Cache for loaded models
_translator = None
_tokenizer_source = None
_tokenizer_target = None
_model_path = None
_device = None

def is_cjk(character):
    """"
    Checks whether character is CJK.

        >>> is_cjk(u'\u33fe')
        True
        >>> is_cjk(u'\uFE5F')
        False

    :param character: The character that needs to be checked.
    :type character: char
    :return: bool
    """
    return any([start <= ord(character) <= end for start, end in 
                [(4352, 4607), (11904, 42191), (43072, 43135), (44032, 55215), 
                 (63744, 64255), (65072, 65103), (65381, 65500), 
                 (131072, 196607)]
                ])

def load_translation_models(model_path=None, device='cpu'):
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
    
    has_cjk = any(is_cjk(c) for c in text)
    if not has_cjk:
        return "", False
    
    translator, tokenizer_source, tokenizer_target = load_translation_models(model_path, device)
    
    # Tokenize
    tokenized = tokenizer_source.encode(text, out_type=str)
    
    # Translate
    translated = translator.translate_batch(source=[tokenized], beam_size=beam_size)
    
    # Decode
    translated_text = tokenizer_target.decode(translated[0].hypotheses[0]).replace('<unk>', '')
    
    return translated_text, True
