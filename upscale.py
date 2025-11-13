#!/usr/bin/env python3
"""
GPU-accelerated image upscaler with multi-model support.
"""

import argparse
import torch
import numpy as np
import subprocess
import sys
from pathlib import Path
from PIL import Image
from models import get_model, MODEL_REGISTRY
from utils import estimate_vram_usage, should_tile, get_available_vram
import time

def tile_process(img_tensor, model, tile_size=512, tile_overlap=32):
    """
    Process large images by splitting into tiles with overlap.
    
    Args:
        img_tensor: Input tensor (1, C, H, W)
        model: Model with __call__ method
        tile_size: Size of each tile in pixels
        tile_overlap: Overlap between tiles to prevent seams
        
    Returns:
        Output tensor
    """
    batch, channels, height, width = img_tensor.shape
    scale = model.scale
    output_height = height * scale
    output_width = width * scale
    
    # Create output tensor on GPU
    output = torch.zeros(
        (batch, channels, output_height, output_width),
        dtype=img_tensor.dtype,
        device=img_tensor.device
    )
    
    # Calculate number of tiles
    stride = tile_size - tile_overlap
    tiles_x = (width + stride - 1) // stride
    tiles_y = (height + stride - 1) // stride
    
    print(f"  Tiling: {tiles_x}x{tiles_y} = {tiles_x * tiles_y} tiles")
    
    # Process each tile
    tile_count = 0
    for y in range(tiles_y):
        for x in range(tiles_x):
            tile_count += 1
            
            # Calculate tile boundaries
            x_start = x * stride
            y_start = y * stride
            x_end = min(x_start + tile_size, width)
            y_end = min(y_start + tile_size, height)
            
            # Extract tile
            tile = img_tensor[:, :, y_start:y_end, x_start:x_end]
            
            # Process tile
            with torch.no_grad():
                tile_output = model(tile)
            
            # Calculate output position
            out_x_start = x_start * scale
            out_y_start = y_start * scale
            out_x_end = x_end * scale
            out_y_end = y_end * scale
            
            # Place tile in output
            output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = tile_output
            
            # Progress indicator
            if tile_count % 10 == 0 or tile_count == tiles_x * tiles_y:
                print(f"    {tile_count}/{tiles_x * tiles_y} tiles...", end='\r')
    
    print()  # New line after progress
    return output


def process_with_model(img_tensor, model, use_fp16, needs_tiling):
    """Process image through a single model"""
    
    # Convert tensor to match model precision
    if hasattr(model, 'using_fp16') and model.using_fp16:
        # Model is FP16, ensure tensor is FP16
        if img_tensor.dtype != torch.float16:
            img_tensor = img_tensor.half()
    else:
        # Model is FP32, ensure tensor is FP32
        if img_tensor.dtype != torch.float32:
            img_tensor = img_tensor.float()
    
    # Perform upscaling
    start_time = time.time()
    if needs_tiling:
        print(f"  Processing with tiling...")
        output_tensor = tile_process(img_tensor, model, tile_size=512, tile_overlap=32)
    else:
        print(f"  Processing (single pass)...")
        with torch.no_grad():
            output_tensor = model(img_tensor)
    
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.2f}s")
    
    return output_tensor, elapsed


def tensor_to_image(tensor):
    """Convert tensor back to PIL Image"""
    output_np = tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    output_np = (output_np * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output_np)


def main():
    parser = argparse.ArgumentParser(
        prog='img-enhance',
        description='''
GPU-Accelerated AI Image Upscaler & Enhancer

Upscale and enhance images using state-of-the-art AI models. This tool uses your
GPU to intelligently reconstruct details, remove artifacts, and improve image quality
beyond simple resizing.

Key Features:
  • Multiple AI models for different use cases (photos, graphics, mixed content)
  • Smart auto-scaling based on input image size
  • FP16 support for 2-3x faster processing on compatible models
  • Tiling system for processing very large images
  • Enhance-only mode for quality improvement without size change
        ''',
        epilog='''
Examples:
  Interactive Mode:
    %(prog)s                                    # Launch guided interactive mode
    %(prog)s --interactive                      # Same as above
  
  Upscale Examples:
    %(prog)s -i photo.jpg                       # Auto-detect scale, use UltraSharp
    %(prog)s -i photo.jpg -m nomos8k --fp16     # Use Nomos8k with FP16 acceleration
    %(prog)s -i small.jpg -s 4 -m all           # 4x upscale with all models for comparison
    %(prog)s -i image.png -m ultrasharp,swinir  # Compare two specific models
  
  Enhance-Only Examples:
    %(prog)s -i blurry.jpg --enhance-only       # Improve quality, keep original size
    %(prog)s -i compressed.jpg -m nomos8k --enhance-only --fp16
  
Model Descriptions:
  ultrasharp  - Sharp details, excellent for text/graphics (Fast, FP16 supported)
  realesrgan  - Balanced results, handles compression well (Fastest, FP16 supported)
  nomos8k     - Best for high-quality photos, realistic textures (Slow, FP32 only)
  swinir      - Complex scenes, mixed content types (Medium speed, FP32 only)
  all         - Run all models for side-by-side comparison

For more information, visit: https://github.com/kevinmcmahonxyz/image-upscaler
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--input', '-i',
        required=True,
        metavar='PATH',
        help='Path to input image file (supports: jpg, png, webp, bmp, tiff)'
    )
    
    # Optional arguments
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument(
        '--output', '-o',
        metavar='PATH',
        help='''Output file path. If not specified, automatically generates filename
in the same directory as input with format: input_MODELNAME_xN.ext
(e.g., photo_ultrasharp_x4.jpg). For enhance-only mode: input_MODELNAME_enhanced.ext'''
    )
    
    optional.add_argument(
        '--scale', '-s',
        type=int,
        choices=[2, 4],
        metavar='N',
        help='''Upscale factor: 2 (double size) or 4 (quadruple size).
If not specified, automatically detects based on image size:
  • Images ≥1000px: uses 2x (conservative for large images)
  • Images <1000px: uses 4x (aggressive for small images)'''
    )
    
    optional.add_argument(
        '--models', '-m',
        default='ultrasharp',
        metavar='MODEL',
        help='''AI model(s) to use for processing. Options:
  • ultrasharp  - ESRGAN-based, sharp edges, good for text/graphics
  • realesrgan  - Original Real-ESRGAN, balanced, handles compression
  • nomos8k     - DAT architecture, best for photorealistic content
  • swinir      - Transformer-based, good for complex/mixed scenes
  • all         - Process with all models for comparison
Use comma-separated list for multiple models (e.g., ultrasharp,nomos8k).
Default: ultrasharp'''
    )
    
    optional.add_argument(
        '--fp16',
        action='store_true',
        help='''Enable half-precision (FP16) processing for 2-3x faster performance.
Compatible with ultrasharp and realesrgan models. Other models will
automatically fall back to FP32. Reduces VRAM usage by ~50%% with
minimal quality impact. Recommended for most use cases.'''
    )
    
    optional.add_argument(
        '--enhance-only',
        action='store_true',
        help='''Enhance image quality without changing dimensions. Processes the image
through a 4x upscaling model, then downsamples back to original size.
This technique removes blur, compression artifacts, and noise while
maintaining the same resolution. Ideal for cleaning up large but
low-quality images. Outputs named: input_MODELNAME_enhanced.ext'''
    )
    
    optional.add_argument(
        '--interactive', '-I',
        action='store_true',
        help='Launch interactive mode with step-by-step prompts (same as running with no arguments)'
    )
    
    args = parser.parse_args()
    
    # Handle interactive mode
    if args.interactive:
        script_dir = Path(__file__).parent
        subprocess.run([sys.executable, str(script_dir / 'interactive.py')])
        return
    
    # Parse model list
    if args.models.lower() == 'all':
        model_names = list(MODEL_REGISTRY.keys())
    else:
        model_names = [m.strip().lower() for m in args.models.split(',')]
    
    # Validate models
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            print(f"Error: Unknown model '{model_name}'")
            print(f"Available models: {', '.join(MODEL_REGISTRY.keys())}")
            return
    
    print(f"=== Multi-Model Image Upscaler ===")
    print(f"Models to run: {', '.join(model_names)}\n")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load image
    img = Image.open(args.input).convert('RGB')
    width, height = img.size
    print(f"Input: {width}x{height}")
    
    # Determine scale factor
    if args.enhance_only:
        scale = 4  # Process through 4x model for quality
        final_scale = 1  # But resize back to original
        print(f"Scale: 4x→1x (enhance-only mode)")
    elif args.scale:
        scale = args.scale
        final_scale = scale
        print(f"Scale: {scale}x (manual)")
    else:
        scale = 2 if (width >= 1000 or height >= 1000) else 4
        final_scale = scale
        print(f"Scale: {scale}x (auto-detected)")
    
    # Memory check (use scale=4 for enhance-only since we process at 4x internally)
    estimated_vram = estimate_vram_usage(width, height, scale)
    available_vram, total_vram = get_available_vram()
    needs_tiling = should_tile(width, height, scale)
    
    print(f"\n=== Memory Info ===")
    print(f"VRAM: {available_vram:.2f}/{total_vram:.2f} GB available")
    print(f"Estimated: {estimated_vram:.2f} GB needed")
    print(f"Tiling: {'Yes' if needs_tiling else 'No'}")
    
    # Convert image to tensor once (reused for all models)
    print(f"\n=== Preparing Image ===")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    if args.fp16:
        img_tensor = img_tensor.half()
        print("Using FP16 precision")
    
    # Process with each model
    results = {}
    
    for model_name in model_names:
        print(f"\n=== Model: {model_name} ===")
        
        # Create and load model
        model = get_model(model_name, device=device)
        model.load(use_fp16=args.fp16)
        
        # Process image
        output_tensor, elapsed = process_with_model(img_tensor, model, args.fp16, needs_tiling)
        
        # Convert to PIL image
        output_img = tensor_to_image(output_tensor)
        
        # Handle scale adjustments
        if final_scale != model.scale:
            # Either: 2x requested with 4x model, OR enhance-only mode
            new_width = width * final_scale
            new_height = height * final_scale
            output_img = output_img.resize((new_width, new_height), Image.LANCZOS)
            
            if args.enhance_only:
                print(f"  Resized back to original: {new_width}x{new_height}")
            else:
                print(f"  Resized from {model.scale}x to {final_scale}x: {new_width}x{new_height}")
        
        # Store result
        results[model_name] = {
            'image': output_img,
            'time': elapsed
        }
    
    # Save outputs
    print(f"\n=== Saving Results ===")
    input_path = Path(args.input)
    
    for model_name, result in results.items():
        # Generate output filename
        if args.output:
            # User provided output path - add model name
            output_path = Path(args.output)
            output_file = output_path.parent / f"{output_path.stem}_{model_name}{output_path.suffix}"
        else:
            # Auto-generate output path
            if args.enhance_only:
                output_file = input_path.parent / f"{input_path.stem}_{model_name}_enhanced{input_path.suffix}"
            else:
                output_file = input_path.parent / f"{input_path.stem}_{model_name}_x{final_scale}{input_path.suffix}"
        
        # Save
        result['image'].save(output_file)
        final_w, final_h = result['image'].size
        print(f"{model_name}: {output_file} ({final_w}x{final_h}, {result['time']:.2f}s)")
    
    print("\n✓ All models completed!")


if __name__ == '__main__':
    main()