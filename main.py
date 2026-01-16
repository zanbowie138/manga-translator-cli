#!/usr/bin/env python3
"""
CLI for manga page translation.

Supports translating single images or entire folders of manga pages.
"""

import argparse
import sys
from pathlib import Path
from src.manga_translate import translate_manga_page, translate_manga_folder


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate manga pages from Japanese to English",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate a single page
  python main.py input/page1.png --output output

  # Translate an entire folder
  python main.py input/ --output output --folder

  # Translate with custom settings
  python main.py input/page1.png --output output --conf-threshold 0.3 --font fonts/custom.ttf

  # Save all intermediate outputs
  python main.py input/ --output output --folder --save-all
        """
    )
    
    # Input/Output
    parser.add_argument(
        'input',
        type=str,
        help='Input image file or folder path'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output folder path (default: output)'
    )
    parser.add_argument(
        '--folder', '-f',
        action='store_true',
        help='Process entire folder instead of single file'
    )
    
    # Detection parameters
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.25,
        help='Confidence threshold for bubble detection (0-1, default: 0.25)'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (0-1, default: 0.45)'
    )
    parser.add_argument(
        '--parent-box-threshold',
        type=int,
        default=10,
        help='Threshold for removing parent boxes (default: 10)'
    )
    
    # Bubble processing
    parser.add_argument(
        '--threshold-value',
        type=int,
        default=200,
        help='Threshold value for bubble mask detection (default: 200)'
    )
    
    # Text parameters
    parser.add_argument(
        '--font',
        type=str,
        default=None,
        help='Path to font file (default: system default)'
    )
    
    # Translation parameters
    parser.add_argument(
        '--translation-model',
        type=str,
        default=None,
        help='Path to translation model (default: auto-download)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for translation (default: cpu)'
    )
    parser.add_argument(
        '--beam-size',
        type=int,
        default=5,
        help='Beam size for translation (default: 5)'
    )
    
    # OCR parameters
    parser.add_argument(
        '--ocr-model',
        type=str,
        default='jzhang533/PaddleOCR-VL-For-Manga',
        help='OCR model ID (default: jzhang533/PaddleOCR-VL-For-Manga)'
    )
    parser.add_argument(
        '--ocr-max-tokens',
        type=int,
        default=2048,
        help='Maximum tokens for OCR (default: 2048)'
    )
    
    # Output toggles
    parser.add_argument(
        '--save-all',
        action='store_true',
        help='Save all intermediate outputs (speech bubbles, interiors, cleaned)'
    )
    parser.add_argument(
        '--save-speech-bubbles',
        action='store_true',
        help='Save annotated detection images'
    )
    parser.add_argument(
        '--save-bubble-interiors',
        action='store_true',
        help='Save blue bubble interior visualizations'
    )
    parser.add_argument(
        '--save-cleaned',
        action='store_true',
        help='Save cleaned images (before text drawing)'
    )
    parser.add_argument(
        '--no-translated',
        action='store_true',
        help='Do not save translated images'
    )
    
    # Processing options
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop processing on first error (folder mode only)'
    )
    
    return parser.parse_args()


def validate_args(args):
    """Validate command-line arguments."""
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}", file=sys.stderr)
        return False
    
    if args.folder:
        if not input_path.is_dir():
            print(f"Error: Input path is not a directory: {args.input}", file=sys.stderr)
            return False
    else:
        if not input_path.is_file():
            print(f"Error: Input path is not a file: {args.input}", file=sys.stderr)
            return False
        
        # Check if it's a supported image format
        supported_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.webp', '.WEBP'}
        if input_path.suffix not in supported_extensions:
            print(f"Error: Unsupported file format: {input_path.suffix}", file=sys.stderr)
            print(f"Supported formats: {', '.join(supported_extensions)}", file=sys.stderr)
            return False
    
    # Validate thresholds
    if not 0 <= args.conf_threshold <= 1:
        print(f"Error: conf-threshold must be between 0 and 1", file=sys.stderr)
        return False
    
    if not 0 <= args.iou_threshold <= 1:
        print(f"Error: iou-threshold must be between 0 and 1", file=sys.stderr)
        return False
    
    return True


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    if not validate_args(args):
        sys.exit(1)
    
    # Determine output toggles
    if args.save_all:
        save_speech_bubbles = True
        save_bubble_interiors = True
        save_cleaned = True
        save_translated = True
    else:
        save_speech_bubbles = args.save_speech_bubbles
        save_bubble_interiors = args.save_bubble_interiors
        save_cleaned = args.save_cleaned
        save_translated = not args.no_translated
    
    verbose = not args.quiet
    
    # Common parameters for both functions
    common_params = {
        'conf_threshold': args.conf_threshold,
        'iou_threshold': args.iou_threshold,
        'parent_box_threshold': args.parent_box_threshold,
        'threshold_value': args.threshold_value,
        'font_path': args.font,
        'translation_model_path': args.translation_model,
        'translation_device': args.device,
        'translation_beam_size': args.beam_size,
        'ocr_model_id': args.ocr_model,
        'ocr_max_new_tokens': args.ocr_max_tokens,
        'save_speech_bubbles': save_speech_bubbles,
        'save_bubble_interiors': save_bubble_interiors,
        'save_cleaned': save_cleaned,
        'save_translated': save_translated,
        'verbose': verbose
    }
    
    try:
        if args.folder:
            # Process folder
            results = translate_manga_folder(
                input_folder=args.input,
                output_folder=args.output,
                continue_on_error=not args.stop_on_error,
                **common_params
            )
            
            # Print summary
            if verbose:
                print(f"\n{'='*50}")
                print("Summary")
                print(f"{'='*50}")
                print(f"Total files: {results['total_files']}")
                print(f"Successfully processed: {results['successful_count']}")
                print(f"Failed: {results['failed_count']}")
        
        else:
            # Process single file
            results = translate_manga_page(
                input_image_path=args.input,
                output_folder=args.output,
                **common_params
            )
            
            # Print summary
            if verbose and results['translated_image']:
                translated_path = results['output_paths'].get('translated')
                if translated_path:
                    print(f"\nTranslation complete! Output saved to: {translated_path}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
