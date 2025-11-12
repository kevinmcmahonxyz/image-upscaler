import torch

def estimate_vram_usage(width, height, scale, bytes_per_pixel=12):
    """
    Estimate VRAM needed for upscaling in GB.
    
    Args:
        width: Input image width
        height: Input image height
        scale: Upscale factor (2 or 4)
        bytes_per_pixel: Memory per pixel (default 12 for FP32 RGB)
    
    Returns:
        Estimated VRAM in GB
    """
    # Output dimensions after upscaling
    output_width = width * scale
    output_height = height * scale
    
    # Memory calculation:
    # - Input tensor: width * height * 3 channels * 4 bytes (FP32)
    # - Output tensor: output_width * output_height * 3 * 4
    # - Intermediate activations: ~3x output size (rough estimate)
    # - Model weights: ~70MB (relatively small, ignored here)
    
    input_memory = width * height * bytes_per_pixel
    output_memory = output_width * output_height * bytes_per_pixel
    intermediate_memory = output_memory * 3  # Model's internal layers
    
    total_bytes = input_memory + output_memory + intermediate_memory
    total_gb = total_bytes / (1024 ** 3)
    
    return total_gb


def get_available_vram():
    """Get available VRAM in GB"""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / (1024 ** 3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
        available = total_memory - allocated_memory
        return available, total_memory
    return 0, 0


def should_tile(width, height, scale, vram_threshold=18):
    """
    Determine if image should be tiled based on size.
    
    Args:
        width: Input image width
        height: Input image height  
        scale: Upscale factor
        vram_threshold: Max VRAM to use in GB (leave buffer for system)
    
    Returns:
        Boolean indicating if tiling is needed
    """
    estimated = estimate_vram_usage(width, height, scale)
    return estimated > vram_threshold