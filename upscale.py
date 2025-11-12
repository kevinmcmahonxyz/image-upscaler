import argparse
import torch
import numpy as np
import os
from pathlib import Path
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader
import urllib.request

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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Upscale images using AI models')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--scale', type=int, choices=[2, 4], help='Upscale factor (default: auto-detect)')
    parser.add_argument('--model', type=str, default='UltraSharp', 
                       choices=['UltraSharp', 'RealESRGAN'], 
                       help='Model to use (default: UltraSharp)')
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
    model.eval()  # Set to evaluation mode
    
    print(f"Model architecture: {model.architecture.name}")
    print(f"Model scale: {model.scale}x")

    # Convert PIL image to tensor
    # PIL: (H, W, C) with values 0-255
    # Tensor: (C, H, W) with values 0-1
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
    img_tensor = img_tensor.to(device)

    # Perform upscaling
    print("Upscaling image...")
    with torch.no_grad():  # Disable gradient calculation for inference
        output_tensor = model(img_tensor)

    # Convert back to PIL image
    # Remove batch dimension, move to CPU, convert to numpy
    output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
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
    print(f"✓ Output saved: {final_width}x{final_height} -> {args.output}")

if __name__ == '__main__':
    main()