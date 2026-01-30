"""Configuration dataclass for manga translation pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration for manga translation pipeline parameters."""

    # Detection parameters
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    parent_box_threshold: int = 20
    bbox_processing: str = 'remove-parent'

    # Bubble processing parameters
    threshold_value: int = 200

    # Text rendering parameters
    font_path: str = 'fonts/CC Astro City Int Regular.ttf'

    # Translation parameters
    translation_model_path: Optional[str] = None
    translation_device: str = 'cpu'
    translation_beam_size: int = 5

    # OCR parameters
    ocr_model_id: str = 'jzhang533/PaddleOCR-VL-For-Manga'
    ocr_max_new_tokens: int = 2048

    # Output toggles
    save_speech_bubbles: bool = False
    save_bubble_interiors: bool = False
    save_cleaned: bool = False
    save_translated: bool = True

    # Processing options
    silent: bool = False
    stop_on_error: bool = False
    batch: bool = False
    batch_amount: Optional[int] = None
