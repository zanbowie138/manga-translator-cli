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

  # Translate multiple files
  python main.py input/page1.png input/page2.png --output output

  # Translate an entire folder
  python main.py input/ --output output

  # Translate multiple folders
  python main.py folder1/ folder2/ --output output

  # Mix files and folders
  python main.py input/page1.png folder1/ folder2/ --output output

  # Translate with custom settings
  python main.py input/page1.png --output output --conf-threshold 0.3 --font fonts/custom.ttf

  # Save all intermediate outputs
  python main.py input/ --output output --save-all
        """
    )
    
    # Input/Output
    parser.add_argument(
        'input',
        type=str,
        nargs='+',
        help='Input image file(s) or folder(s) path(s)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output folder path (default: output)'
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
        default=20,
        help='Threshold for processing compound speech bubbles (default: 20)'
    )
    parser.add_argument(
        '--bbox-processing',
        type=str,
        default='remove-parent',
        choices=['remove-parent', 'combine-children', 'none'],
        help='Compound speech bubble processing mode: remove-parent (remove parent boxes, keep children), combine-children (combine overlapping/touching compound bubbles), or none (no processing) (default: remove-parent)'
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
        help='Stop processing on first error (applies to folders)'
    )
    
    return parser.parse_args()


def validate_args(args):
    """Validate command-line arguments."""
    supported_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.webp', '.WEBP'}
    all_valid = True
    
    for input_str in args.input:
        input_path = Path(input_str)
        
        if not input_path.exists():
            print(f"Error: Input path does not exist: {input_str}", file=sys.stderr)
            all_valid = False
            continue
        
        if input_path.is_dir():
            # It's a folder, which is valid
            continue
        elif input_path.is_file():
            # It's a file, check if it's a supported image format
            if input_path.suffix not in supported_extensions:
                print(f"Error: Unsupported file format: {input_path.suffix} for {input_str}", file=sys.stderr)
                print(f"Supported formats: {', '.join(supported_extensions)}", file=sys.stderr)
                all_valid = False
        else:
            print(f"Error: Input path is neither a file nor a directory: {input_str}", file=sys.stderr)
            all_valid = False
    
    # Validate thresholds
    if not 0 <= args.conf_threshold <= 1:
        print(f"Error: conf-threshold must be between 0 and 1", file=sys.stderr)
        all_valid = False
    
    if not 0 <= args.iou_threshold <= 1:
        print(f"Error: iou-threshold must be between 0 and 1", file=sys.stderr)
        all_valid = False
    
    return all_valid


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
        'bbox_processing': args.bbox_processing,
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
        supported_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.webp', '.WEBP'}
        total_processed = 0
        total_failed = 0
        all_results = []
        
        for input_str in args.input:
            input_path = Path(input_str)
            
            if input_path.is_dir():
                # Process folder
                if verbose:
                    print(f"\n{'='*50}")
                    print(f"Processing folder: {input_str}")
                    print(f"{'='*50}")
                
                results = translate_manga_folder(
                    input_folder=input_str,
                    output_folder=args.output,
                    continue_on_error=not args.stop_on_error,
                    **common_params
                )
                
                total_processed += results['successful_count']
                total_failed += results['failed_count']
                all_results.append(('folder', input_str, results))
                
                if verbose:
                    print(f"\nFolder '{input_str}' summary:")
                    print(f"  Total files: {results['total_files']}")
                    print(f"  Successfully processed: {results['successful_count']}")
                    print(f"  Failed: {results['failed_count']}")
            
            elif input_path.is_file() and input_path.suffix in supported_extensions:
                # Process single file
                if verbose:
                    print(f"\n{'='*50}")
                    print(f"Processing file: {input_str}")
                    print(f"{'='*50}")
                
                results = translate_manga_page(
                    input_image_path=input_str,
                    output_folder=args.output,
                    **common_params
                )
                
                if results.get('translated_image'):
                    total_processed += 1
                    all_results.append(('file', input_str, results))
                    
                    if verbose:
                        translated_path = results['output_paths'].get('translated')
                        if translated_path:
                            print(f"\nFile '{input_str}' processed successfully!")
                            print(f"  Output: {translated_path}")
                else:
                    total_failed += 1
                    if verbose:
                        print(f"\nFile '{input_str}' failed to process")
        
        # Print overall summary if multiple inputs
        if len(args.input) > 1 and verbose:
            print(f"\n{'='*50}")
            print("Overall Summary")
            print(f"{'='*50}")
            print(f"Total inputs: {len(args.input)}")
            print(f"Successfully processed: {total_processed}")
            print(f"Failed: {total_failed}")
    
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
