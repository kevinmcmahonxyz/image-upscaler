#!/usr/bin/env python3
"""
Interactive CLI for image upscaling with guided prompts.
"""

import sys
from pathlib import Path
from models import MODEL_REGISTRY


def print_header():
    """Print welcome header"""
    print("\n" + "="*50)
    print("   Image Upscaler - Interactive Mode")
    print("="*50 + "\n")


def get_input_path():
    """Prompt for and validate input image path"""
    while True:
        path_input = input("Enter image path: ").strip()
        
        if not path_input:
            print("  ❌ Path cannot be empty")
            continue
        
        path = Path(path_input)
        
        if not path.exists():
            print(f"  ❌ File not found: {path}")
            continue
        
        if not path.is_file():
            print(f"  ❌ Not a file: {path}")
            continue
        
        # Check if it's an image (basic check)
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        if path.suffix.lower() not in valid_extensions:
            print(f"  ⚠️  Warning: {path.suffix} may not be a valid image format")
            confirm = input("  Continue anyway? [y/N]: ").strip().lower()
            if confirm != 'y':
                continue
        
        print(f"  ✓ Found: {path}")
        return str(path)


def get_task():
    """Prompt for task selection"""
    print("\nSelect task:")
    print("  [1] Upscale & Enhance (make larger with AI enhancement)")
    print("  [2] Enhance Only (improve quality, keep original size)")
    
    while True:
        choice = input("\nChoice [1-2]: ").strip()
        
        if choice == '1':
            return 'upscale'
        elif choice == '2':
            return 'enhance'
        else:
            print("  ❌ Invalid choice. Enter 1 or 2")


def get_scale_factor(task):
    """Prompt for scale factor (only if upscaling)"""
    if task == 'enhance':
        return None  # Enhance-only doesn't need scale
    
    print("\nSelect upscale factor:")
    print("  [1] 2x (double size)")
    print("  [2] 4x (quadruple size)")
    print("  [3] Auto-detect (4x for small images, 2x for large)")
    
    while True:
        choice = input("\nChoice [1-3]: ").strip()
        
        if choice == '1':
            return 2
        elif choice == '2':
            return 4
        elif choice == '3':
            return None  # None means auto-detect
        else:
            print("  ❌ Invalid choice. Enter 1, 2, or 3")


def get_models():
    """Prompt for model selection"""
    print("\nSelect model(s):")
    print("  [1] ultrasharp     - Sharp details, good for text/graphics (Fast, FP16)")
    print("  [2] realesrgan     - Balanced, handles compression well (Fastest, FP16)")
    print("  [3] nomos8k        - Best for photos, realistic textures (Slow, FP32)")
    print("  [4] swinir         - Complex scenes, mixed content (Medium, FP32)")
    print("  [5] ALL models     - Run all for A/B comparison")
    
    model_map = {
        '1': 'ultrasharp',
        '2': 'realesrgan',
        '3': 'nomos8k',
        '4': 'swinir',
        '5': 'all'
    }
    
    while True:
        choice = input("\nEnter numbers separated by commas (e.g., 1,3): ").strip()
        
        if not choice:
            print("  ❌ Please select at least one model")
            continue
        
        # Parse selections
        selections = [s.strip() for s in choice.split(',')]
        
        # Check if '5' (all) is selected
        if '5' in selections:
            print("  ✓ Selected: ALL models")
            return 'all'
        
        # Validate all selections
        invalid = [s for s in selections if s not in model_map]
        if invalid:
            print(f"  ❌ Invalid selection(s): {', '.join(invalid)}")
            continue
        
        # Convert to model names
        selected_models = [model_map[s] for s in selections]
        print(f"  ✓ Selected: {', '.join(selected_models)}")
        return ','.join(selected_models)


def get_fp16_preference():
    """Prompt for FP16 usage"""
    print("\nUse FP16 for faster processing?")
    print("  (Note: Some models don't support FP16 and will fall back to FP32)")
    
    while True:
        choice = input("Use FP16? [Y/n]: ").strip().lower()
        
        if choice in ['', 'y', 'yes']:
            print("  ✓ FP16 enabled")
            return True
        elif choice in ['n', 'no']:
            print("  ✓ FP16 disabled")
            return False
        else:
            print("  ❌ Please enter Y or N")


def build_command(input_path, task, scale, models, use_fp16):
    """Build the upscale.py command from user selections"""
    cmd = ['python', 'upscale.py', '--input', input_path]
    
    # Add models
    cmd.extend(['--models', models])
    
    # Add task-specific flags
    if task == 'enhance':
        cmd.append('--enhance-only')
    
    # Add scale if specified (only for upscale task)
    if task == 'upscale' and scale is not None:
        cmd.extend(['--scale', str(scale)])
    
    # Add FP16 if enabled
    if use_fp16:
        cmd.append('--fp16')
    
    return cmd


def main():
    """Main interactive CLI flow"""
    print_header()
    
    # Gather all inputs
    input_path = get_input_path()
    task = get_task()
    scale = get_scale_factor(task)
    models = get_models()
    use_fp16 = get_fp16_preference()
    
    # Build and execute command
    print("\n" + "="*50)
    print("Processing...")
    print("="*50 + "\n")
    
    cmd = build_command(input_path, task, scale, models, use_fp16)
    
    import subprocess
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "="*50)
        print("✓ Complete!")
        print("="*50 + "\n")
    else:
        print("\n" + "="*50)
        print("❌ Processing failed")
        print("="*50 + "\n")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)