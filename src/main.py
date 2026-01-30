#!/usr/bin/env python3
"""
CLI for manga page translation.

Supports translating single images or entire folders of manga pages.
"""

import argparse
import sys
import time
from pathlib import Path
from .manga_translate import translate_manga_page, translate_manga_folder, translate_manga_page_batch, translate_manga_folder_batch
from .config import Config
from .console import Console
from .output_manager import OutputManager


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
        default='fonts/CC Astro City Int Regular.ttf',
        help='Path to font file (default: fonts/CC Astro City Int Regular.ttf)'
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
    
    # Batch processing
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Enable batch processing mode (process all pages through each pipeline step together)'
    )
    parser.add_argument(
        '--batch-amount',
        type=int,
        default=None,
        help='Maximum number of pages to process in each batch (default: None = process all at once)'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Measure and display processing time'
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

    # Create configuration
    config = Config(
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        parent_box_threshold=args.parent_box_threshold,
        bbox_processing=args.bbox_processing,
        threshold_value=args.threshold_value,
        font_path=args.font,
        translation_model_path=args.translation_model,
        translation_device=args.device,
        translation_beam_size=args.beam_size,
        ocr_model_id=args.ocr_model,
        ocr_max_new_tokens=args.ocr_max_tokens,
        save_speech_bubbles=save_speech_bubbles,
        save_bubble_interiors=save_bubble_interiors,
        save_cleaned=save_cleaned,
        save_translated=save_translated,
        silent=args.quiet,
        stop_on_error=args.stop_on_error,
        batch=args.batch,
        batch_amount=args.batch_amount
    )

    # Create console handler
    console = Console(quiet=args.quiet)
    
    try:
        # Start benchmark timer if enabled
        start_time = time.time() if args.benchmark else None

        supported_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.webp', '.WEBP'}
        total_processed = 0
        total_failed = 0
        all_results = []

        # Collect all images for batch processing (if enabled)
        all_batch_images = []

        if args.batch:
            # Collect all images from folders and individual files
            for input_str in args.input:
                input_path = Path(input_str)

                if input_path.is_dir():
                    # Collect all images from this folder
                    image_files = [
                        str(f) for f in input_path.iterdir()
                        if f.is_file() and f.suffix in supported_extensions
                    ]
                    image_files.sort(key=lambda x: Path(x).name)
                    all_batch_images.extend(image_files)

                    console.info(f"Collected {len(image_files)} image(s) from folder: {input_str}")

                elif input_path.is_file() and input_path.suffix in supported_extensions:
                    all_batch_images.append(input_str)

            # Process all collected images together in batch mode
            if all_batch_images:
                console.section(f"Batch processing {len(all_batch_images)} image(s) from all inputs...")

                try:
                    results_dict = translate_manga_page_batch(
                        input_image_paths=all_batch_images,
                        output_folder=args.output,
                        config=config,
                        console=console
                    )

                    # Process results
                    for input_str in all_batch_images:
                        results = results_dict.get(input_str, {})
                        if results.get('translated_image'):
                            total_processed += 1
                            all_results.append(('file', input_str, results))
                        else:
                            total_failed += 1

                    console.success(f"Batch processing complete: {total_processed} successful, {total_failed} failed")
                except Exception as e:
                    console.error(f"Error in batch processing: {e}")
                    if args.stop_on_error:
                        raise
                    total_failed = len(all_batch_images)
            else:
                console.info("No images found to process")
        else:
            # Non-batch mode: process each input separately
            for input_str in args.input:
                input_path = Path(input_str)

                if input_path.is_dir():
                    # Process folder
                    console.section(f"Processing folder: {input_str}")

                    results = translate_manga_folder(
                        input_folder=input_str,
                        output_folder=args.output,
                        config=config,
                        console=console
                    )

                    total_processed += results['successful_count']
                    total_failed += results['failed_count']
                    all_results.append(('folder', input_str, results))

                    console.print(f"\nFolder '{input_str}' summary:")
                    console.print(f"  Total files: {results['total_files']}")
                    console.print(f"  Successfully processed: {results['successful_count']}")
                    console.print(f"  Failed: {results['failed_count']}")

                elif input_path.is_file() and input_path.suffix in supported_extensions:
                    # Process single file
                    console.section(f"Processing file: {input_str}")

                    results = translate_manga_page(
                        input_image_path=input_str,
                        output_folder=args.output,
                        config=config,
                        console=console
                    )

                    if results.get('translated_image'):
                        total_processed += 1
                        all_results.append(('file', input_str, results))

                        translated_path = results['output_paths'].get('translated')
                        if translated_path:
                            console.success(f"File '{input_str}' processed successfully!")
                            console.print(f"  Output: {translated_path}")
                    else:
                        total_failed += 1
                        console.error(f"File '{input_str}' failed to process")
        
        # Print overall summary if multiple inputs
        if len(args.input) > 1:
            console.section("Overall Summary")
            console.print(f"Total inputs: {len(args.input)}")
            console.print(f"Successfully processed: {total_processed}")
            console.print(f"Failed: {total_failed}")

        # Print benchmark results if enabled
        if args.benchmark and start_time is not None:
            from rich.table import Table

            end_time = time.time()
            elapsed_time = end_time - start_time

            table = Table(title="Benchmark Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total processing time", f"{elapsed_time:.2f} seconds")
            if total_processed > 0:
                avg_time_per_page = elapsed_time / total_processed
                table.add_row("Average time per page", f"{avg_time_per_page:.2f} seconds")
            table.add_row("Total pages processed", str(total_processed))

            console.console.print(table)
    
    except KeyboardInterrupt:
        console.error("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        console.error(f"Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
