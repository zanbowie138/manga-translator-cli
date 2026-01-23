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
- UV package manager (highly recommended) or pip

### Installing UV

If you don't have `uv` installed, you can install it using one of the following methods:

**Using pip:**
```bash
pip install uv
```

**Using the official installer (recommended):**
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Using Homebrew (macOS):**
```bash
brew install uv
```

**Using cargo:**
```bash
cargo install uv
```

For more installation options, visit: https://github.com/astral-sh/uv

### Setup

You can install this project in two ways:

#### Option 1: Install as a tool (recommended)

This installs the `manga-translate` command globally:

**For CPU-only:**
```bash
uv tool install . --from 'torch' 
```

**For CUDA support (GPU acceleration):**
```bash
uv tool install --from '. [cu130]' .
```

#### Option 2: Development installation

For development or if you want to run from the source directory:

1. Clone or download this repository

2. Install dependencies using UV:

   **For CPU-only:**
   ```bash
   uv sync --extra cpu
   ```

   **For CUDA support (GPU acceleration):**
   ```bash
   uv sync --extra cu130
   ```

   Note: CPU versions of PyTorch are installed by default. Use the `--extra cu130` flag to install CUDA 13.0 versions for GPU acceleration.

### Models

Models will be automatically downloaded on first use:
- YOLO model for bubble detection
- PaddleOCR-VL model for text extraction
- Sugoi-v4 model for translation

## Usage

### Single Image Translation

```bash
manga-translate input/page1.png --output output
```

### Folder Translation

```bash
manga-translate input/ --output output
```

### Common Options

Save all intermediate outputs:
```bash
manga-translate input/page1.png --output output --save-all
```

Use custom font:
```bash
manga-translate input/page1.png --output output --font "fonts/CC Astro City Int Regular.ttf"
```

Adjust detection sensitivity:
```bash
manga-translate input/page1.png --output output --conf-threshold 0.3 --iou-threshold 0.5
```

Use GPU acceleration:
```bash
manga-translate input/page1.png --output output --device cuda
```

Quiet mode:
```bash
manga-translate input/page1.png --output output --quiet
```

### Available Options

- `--output, -o`: Output folder path (default: output)
- `--folder, -f`: Process entire folder instead of single file
- `--conf-threshold`: Confidence threshold for bubble detection (0-1, default: 0.25)
- `--iou-threshold`: IoU threshold for NMS (0-1, default: 0.45)
- `--font`: Path to font file for translated text
- `--device`: Device for OCR and translation (cpu or cuda, default: cpu). Controls which device is used for both text extraction and translation.
- `--save-all`: Save all intermediate outputs
- `--save-speech-bubbles`: Save annotated detection images
- `--save-bubble-interiors`: Save bubble interior visualizations
- `--save-cleaned`: Save cleaned images before text drawing
- `--quiet, -q`: Suppress progress messages
- `--stop-on-error`: Stop processing on first error (folder mode)

For complete list of options:
```bash
manga-translate --help
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
- GPU support available for faster processing (install with `uv sync --extra cu130` and use `--device cuda`)
- The `--device` argument controls both OCR and translation device usage
- Supported image formats: PNG, JPG, JPEG, WEBP

