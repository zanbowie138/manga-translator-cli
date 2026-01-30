# Manga Translation CLI

Automated tool for translating manga pages from Japanese to English. Detects speech bubbles, extracts Japanese text, translates to English, and renders the translated text back onto the image.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Installing UV](#installing-uv)
  - [Setup](#setup)
  - [Models](#models)
- [Usage](#usage)
  - [Single Image Translation](#single-image-translation)
  - [Folder Translation](#folder-translation)
  - [Common Options](#common-options)
  - [Available Options](#available-options)
- [Output Structure](#output-structure)
- [Dependencies](#dependencies)
- [How It Works](#how-it-works)
- [Switching Between CPU and CUDA](#switching-between-cpu-and-cuda)
- [Notes](#notes)

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
- UV package manager

For uv installation, visit: https://github.com/astral-sh/uv

### Setup

1. Clone or download this repository

2. **Select CPU or CUDA backend** (edit `pyproject.toml`):
   - Open `pyproject.toml`
   - Find `[tool.uv.sources]` section (line ~37)
   - Comment out CPU lines and uncomment CUDA lines for GPU
   - Comment out CUDA lines and uncomment CPU lines for CPU-only
   - Default: CUDA 12.8

3. **Install dependencies:**
   ```bash
   uv sync
   ```

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

## Switching Between CPU and CUDA

After installation, to change PyTorch backend:
1. Edit `pyproject.toml` `[tool.uv.sources]` section (comment/uncomment lines)
2. Run `uv sync`
3. Use `--device cuda` when running to use GPU

## Notes

- First run downloads models (several minutes)
- Translation quality depends on text clarity and font style
- GPU requires CUDA 12.8-compatible NVIDIA GPU + drivers
- `--device` controls both OCR and translation device
- Supported formats: PNG, JPG, JPEG, WEBP

