# Manga Translation Tool

Automated tool for translating manga pages from Japanese to English. Detects speech bubbles, extracts Japanese text, translates to English, and renders the translated text back onto the image.

## Features

- Automatic speech bubble detection using YOLO
- Japanese text extraction using PaddleOCR-VL
- Japanese to English translation using Sugoi-v4
- Text alignment within speech bubble shapes
- Batch processing support for entire folders
- Configurable detection and processing parameters

## Installation

### Requirements

- Python 3.13 or higher
- UV package manager (recommended) or pip

### Setup

1. Clone or download this repository

2. Install dependencies using UV:
```bash
uv sync
```

3. Models will be automatically downloaded on first use:
   - YOLO model for bubble detection
   - PaddleOCR-VL model for text extraction
   - Sugoi-v4 model for translation

## Usage

### Single Image Translation

```bash
python main.py input/page1.png --output output
```

### Folder Translation

```bash
python main.py input/ --output output --folder
```

### Common Options

Save all intermediate outputs:
```bash
python main.py input/page1.png --output output --save-all
```

Use custom font:
```bash
python main.py input/page1.png --output output --font "fonts/CC Astro City Int Regular.ttf"
```

Adjust detection sensitivity:
```bash
python main.py input/page1.png --output output --conf-threshold 0.3 --iou-threshold 0.5
```

Quiet mode:
```bash
python main.py input/page1.png --output output --quiet
```

### Available Options

- `--output, -o`: Output folder path (default: output)
- `--folder, -f`: Process entire folder instead of single file
- `--conf-threshold`: Confidence threshold for bubble detection (0-1, default: 0.25)
- `--iou-threshold`: IoU threshold for NMS (0-1, default: 0.45)
- `--font`: Path to font file for translated text
- `--device`: Device for translation (cpu or cuda, default: cpu)
- `--save-all`: Save all intermediate outputs
- `--save-speech-bubbles`: Save annotated detection images
- `--save-bubble-interiors`: Save bubble interior visualizations
- `--save-cleaned`: Save cleaned images before text drawing
- `--quiet, -q`: Suppress progress messages
- `--stop-on-error`: Stop processing on first error (folder mode)

For complete list of options:
```bash
python main.py --help
```

## Output Structure

When processing files, outputs are organized in subdirectories:

- `translated/`: Final translated images
- `speech_bubbles/`: Annotated images with detected bubbles
- `bubble_interiors/`: Visualization of bubble interiors
- `cleaned/`: Images with bubbles filled (before text rendering)

## Dependencies

- ultralytics: YOLO model for bubble detection
- transformers: PaddleOCR-VL for text extraction
- ctranslate2: Fast translation inference
- sentencepiece: Text tokenization
- torch: Deep learning framework
- opencv-python: Image processing
- pillow: Image manipulation

## How It Works

1. Loads a YOLO model to detect speech bubbles in the image
2. Filters out parent boxes that contain smaller child boxes
3. For each bubble, extracts Japanese text using PaddleOCR-VL
4. Detects if text contains Japanese characters
5. Translates Japanese text to English using Sugoi-v4
6. Cleans bubble interiors by filling with base color
7. Renders translated text within bubble shapes using binary search for optimal font size
8. Saves the final translated image

## Notes

- First run will download models, which may take several minutes
- Translation quality depends on text clarity and font style
- GPU support available for faster translation (use --device cuda)
- Supported image formats: PNG, JPG, JPEG, WEBP

