import ctranslate2
import sentencepiece
from huggingface_hub import snapshot_download
import os
from tqdm import tqdm
import torch

# Cache for loaded models
_cached_translator = None
_cached_tokenizer_source = None
_cached_tokenizer_target = None
_cached_key = None

def _validate_device(device):
    """Validate device availability."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device

def _normalize_model_path(model_path):
    """Normalize model path for cache comparison."""
    if model_path is None:
        return None
    expanded = os.path.expanduser(model_path)
    return os.path.normpath(os.path.abspath(expanded))

def _get_cache_key(model_path, device):
    """Create cache key tuple."""
    return (_normalize_model_path(model_path), device)

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

def load_translation_models(model_path=None, device='cpu', silent=False):
    """Load translation models lazily (only once)"""
    global _cached_translator, _cached_tokenizer_source, _cached_tokenizer_target, _cached_key

    device = _validate_device(device)

    if model_path is None:
        model_path = os.path.join(os.path.expanduser("~"), ".manga-translate", "sugoi-v4-ja-en-ctranslate2")

    cache_key = _get_cache_key(model_path, device)

    # Download model if not exists
    expanded_path = os.path.expanduser(model_path)
    if not os.path.exists(expanded_path):
        if not silent:
            print(f"Downloading translation model to: {model_path}")
        os.makedirs(os.path.dirname(expanded_path), exist_ok=True)
        snapshot_download(repo_id='entai2965/sugoi-v4-ja-en-ctranslate2', local_dir=expanded_path)
        if not silent:
            print(f"Translation model downloaded to: {model_path}")

    # Return cached models if already loaded
    if _cached_translator is not None and cache_key == _cached_key:
        if not silent:
            print(f"Using cached translation model from: {model_path} on {device}")
        return _cached_translator, _cached_tokenizer_source, _cached_tokenizer_target

    # Load models
    if not silent:
        print(f"Loading translation model from: {model_path} on {device}")
    sentencepiece_model_path = os.path.join(expanded_path, 'spm')

    _cached_translator = ctranslate2.Translator(expanded_path, device=device)
    _cached_tokenizer_source = sentencepiece.SentencePieceProcessor(
        os.path.join(sentencepiece_model_path, 'spm.ja.nopretok.model')
    )
    _cached_tokenizer_target = sentencepiece.SentencePieceProcessor(
        os.path.join(sentencepiece_model_path, 'spm.en.nopretok.model')
    )

    _cached_key = cache_key

    if not silent:
        print(f"Translation model loaded successfully")

    return _cached_translator, _cached_tokenizer_source, _cached_tokenizer_target

def translate_phrase(text, model_path=None, device='cpu', beam_size=5, silent=False):
    """
    Translate a Japanese phrase to English

    Args:
        text: Japanese text string to translate
        model_path: Path to model directory (defaults to '~/.manga-translate/sugoi-v4-ja-en-ctranslate2')
        device: Device to use ('cpu' or 'cuda')
        beam_size: Beam size for translation (default 5)
        silent: If True, suppress progress messages

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

    translator, tokenizer_source, tokenizer_target = load_translation_models(model_path, device, silent=silent)

    # Tokenize
    tokenized = tokenizer_source.encode(text, out_type=str)

    # Translate
    translated = translator.translate_batch(source=[tokenized], beam_size=beam_size)

    # Decode
    translated_text = tokenizer_target.decode(translated[0].hypotheses[0]).replace('<unk>', '')

    return translated_text, True


def translate_batch(
    texts: list,
    model_path=None,
    device='cpu',
    beam_size=5,
    silent=False
):
    """
    Batch translate a list of Japanese texts to English.
    
    Filters texts for CJK characters, tokenizes, translates in batch, and decodes results.
    
    Args:
        texts: List of text strings to translate
        model_path: Path to model directory (defaults to '~/.manga-translate/sugoi-v4-ja-en-ctranslate2')
        device: Device to use ('cpu' or 'cuda')
        beam_size: Beam size for translation (default 5)
        silent: If True, suppress progress messages
    
    Returns:
        List of translated text strings (empty string for texts without CJK or failed translations)
    """
    if not texts:
        return []
    
    # Filter texts with CJK characters
    texts_to_translate = []
    text_indices = []  # Track original indices
    
    for idx, text in enumerate(texts):
        if text and text.strip() and any(is_cjk(c) for c in text):
            texts_to_translate.append(text)
            text_indices.append(idx)
    
    if not texts_to_translate:
        return [""] * len(texts)

    # Load translation models
    translator, tokenizer_source, tokenizer_target = load_translation_models(model_path, device, silent=silent)

    # Tokenize all texts with progress bar
    tokenized_texts = []
    valid_indices = []
    iterator = tqdm(zip(text_indices, texts_to_translate), total=len(texts_to_translate),
                   desc="Tokenizing", disable=silent, unit="text")

    for idx, text in iterator:
        try:
            tokenized = tokenizer_source.encode(text, out_type=str)
            tokenized_texts.append(tokenized)
            valid_indices.append(idx)
        except Exception as e:
            if not silent:
                print(f"Warning: Error tokenizing text at index {idx}: {e}")

    if not tokenized_texts:
        return [""] * len(texts)

    # Batch translate
    try:
        if not silent:
            # Show a simple message for the actual translation step (happens quickly)
            print(f"Translating {len(tokenized_texts)} texts...")
        translated_results = translator.translate_batch(
            source=tokenized_texts,
            beam_size=beam_size
        )
    except Exception as e:
        if not silent:
            print(f"Error during batch translation: {e}")
        return [""] * len(texts)

    # Initialize result list with empty strings
    translated_texts = [""] * len(texts)

    # Decode translations and map back to original indices
    for idx, translated_result in zip(valid_indices, translated_results):
        try:
            translated_text = tokenizer_target.decode(translated_result.hypotheses[0]).replace('<unk>', '')
            translated_texts[idx] = translated_text
        except Exception as e:
            if not silent:
                print(f"Warning: Error decoding translation at index {idx}: {e}")

    return translated_texts


def translate_individual(
    texts: list,
    model_path=None,
    device='cpu',
    beam_size=5,
    silent=False
):
    """
    Translate a list of Japanese texts to English individually (not batched).
    
    Processes each text separately, one at a time. Filters texts for CJK characters before processing.
    
    Args:
        texts: List of text strings to translate
        model_path: Path to model directory (defaults to '~/.manga-translate/sugoi-v4-ja-en-ctranslate2')
        device: Device to use ('cpu' or 'cuda')
        beam_size: Beam size for translation (default 5)
        silent: If True, suppress progress messages
    
    Returns:
        List of translated text strings (empty string for texts without CJK or failed translations)
    """
    if not texts:
        return []
    
    # Filter texts with CJK characters
    texts_to_translate = []
    text_indices = []  # Track original indices
    
    for idx, text in enumerate(texts):
        if text and text.strip() and any(is_cjk(c) for c in text):
            texts_to_translate.append(text)
            text_indices.append(idx)
    
    if not texts_to_translate:
        return [""] * len(texts)

    # Load translation models once
    translator, tokenizer_source, tokenizer_target = load_translation_models(model_path, device, silent=silent)

    # Initialize result list with empty strings
    translated_texts = [""] * len(texts)

    # Process each text individually with progress bar
    iterator = tqdm(zip(text_indices, texts_to_translate), total=len(texts_to_translate),
                   desc="Translating", disable=silent, unit="text")

    for idx, text in iterator:
        try:
            # Tokenize
            tokenized = tokenizer_source.encode(text, out_type=str)

            # Translate (single text, but still use translate_batch API)
            translated_results = translator.translate_batch(
                source=[tokenized],
                beam_size=beam_size
            )

            # Decode
            translated_text = tokenizer_target.decode(translated_results[0].hypotheses[0]).replace('<unk>', '')
            translated_texts[idx] = translated_text

        except Exception as e:
            if not silent:
                print(f"Warning: Error translating text at index {idx}: {e}")

    return translated_texts
