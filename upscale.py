import argparse
import torch
import numpy as np
import os
from pathlib import Path
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader
import urllib.request
from utils import estimate_vram_usage, should_tile, get_available_vram
import time  # Add this for performance timing

def download_model(url, model_name):
    """Download model if it doesn't exist locally"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / f"{model_name}.pth"
    
    if model_path.exists():
        print(f"Model already downloaded: {model_path}")
        return str(model_path)
    
    print(f"Downloading {model_name} model...")
    urllib.request.urlretrieve(url, model_path)
    print(f"Model saved to: {model_path}")
    
    return str(model_path)

def tile_process(img_tensor, model, tile_size=512, tile_overlap=32):
    """
    Process large images by splitting into tiles with overlap.
    
    Args:
        img_tensor: Input tensor (1, C, H, W)
        model: Spandrel model
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
    
    print(f"Tiling: {tiles_x}x{tiles_y} = {tiles_x * tiles_y} tiles (tile_size={tile_size}, overlap={tile_overlap})")
    
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
            
            # Blend tile into output (handle overlap)
            if tile_overlap > 0 and (x > 0 or y > 0):
                # Simple blending: use the tile output directly
                # More sophisticated: weighted blending in overlap regions
                output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = tile_output
            else:
                output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = tile_output
            
            # Progress indicator
            if tile_count % 10 == 0 or tile_count == tiles_x * tiles_y:
                print(f"  Processed {tile_count}/{tiles_x * tiles_y} tiles...")
    
    return output

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Upscale images using AI models')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--scale', type=int, choices=[2, 4], help='Upscale factor (default: auto-detect)')
    parser.add_argument('--model', type=str, default='UltraSharp', 
                       choices=['UltraSharp', 'RealESRGAN'], 
                       help='Model to use (default: UltraSharp)')
    parser.add_argument('--fp16', action='store_true', 
                       help='Use half precision (FP16) for faster processing')
    args = parser.parse_args()

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load image to check dimensions
    img = Image.open(args.input).convert('RGB')
    width, height = img.size
    print(f"Input image size: {width}x{height}")

    # Intelligent scale factor selection
    if args.scale:
        scale = args.scale
        print(f"Using manual scale: {scale}x")
    else:
        # Auto-detect appropriate scale based on input size
        if width >= 1000 or height >= 1000:
            scale = 2  # Conservative for large images
            print(f"Large image detected, using 2x upscale")
        else:
            scale = 4  # Aggressive for small images
            print(f"Small image detected, using 4x upscale")

    # Model URLs
    model_configs = {
        'UltraSharp': {
            'url': 'https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth',
            'name': '4x-UltraSharp'
        },
        'RealESRGAN': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'name': 'RealESRGAN_x4plus'
        }
    }

    # Download model if needed
    config = model_configs[args.model]
    model_path = download_model(config['url'], config['name'])
    
    # Load model with spandrel
    print(f"Loading {args.model} model...")
    model = ModelLoader().load_from_file(model_path)
    assert isinstance(model, ImageModelDescriptor)
    
    # Move model to GPU
    model = model.to(device)

    # Convert to FP16 if requested
    if args.fp16:
        model = model.half()
        print("Using FP16 (half precision)")

    model.eval()  # Set to evaluation mode
    
    print(f"Model architecture: {model.architecture.name}")
    print(f"Model scale: {model.scale}x")

    # Convert PIL image to tensor
    # PIL: (H, W, C) with values 0-255
    # Tensor: (C, H, W) with values 0-1
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
    img_tensor = img_tensor.to(device)

    # Convert to FP16 if using half precision
    if args.fp16:
        img_tensor = img_tensor.half()

    # Check if tiling is needed
    needs_tiling = should_tile(width, height, model.scale)
    estimated_vram = estimate_vram_usage(width, height, model.scale)
    available_vram, total_vram = get_available_vram()
    
    print(f"\n=== Memory Info ===")
    print(f"Available VRAM: {available_vram:.2f}/{total_vram:.2f} GB")
    print(f"Estimated need: {estimated_vram:.2f} GB")
    print(f"Tiling: {'Yes' if needs_tiling else 'No'}")
    print()
    
    # Perform upscaling
    start_time = time.time()
    if needs_tiling:
        print("Upscaling with tiling...")
        output_tensor = tile_process(img_tensor, model, tile_size=512, tile_overlap=32)
    else:
        print("Upscaling (single pass)...")
        with torch.no_grad():
            output_tensor = model(img_tensor)
    
    elapsed = time.time() - start_time
    print(f"Processing time: {elapsed:.2f}s")

    # Convert back to PIL image
    # Remove batch dimension, move to CPU, convert to numpy
    output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()  # Add .float() to handle FP16
    output_np = (output_np * 255.0).clip(0, 255).astype(np.uint8)
    output_img = Image.fromarray(output_np)

    # If we want 2x but model is 4x, resize down
    if scale == 2 and model.scale == 4:
        new_width = width * 2
        new_height = height * 2
        output_img = output_img.resize((new_width, new_height), Image.LANCZOS)
        print(f"Resized from 4x to 2x: {new_width}x{new_height}")

    # Save result
    output_img.save(args.output)
    
    final_width, final_height = output_img.size
    print(f"\n=== Results ===")
    print(f"Input: {width}x{height}")
    print(f"Output: {final_width}x{final_height}")
    print(f"Effective scale: {final_width/width:.1f}x")
    print(f"Processing time: {elapsed:.2f}s")
    print(f"Output saved: {args.output}")

if __name__ == '__main__':
    main()