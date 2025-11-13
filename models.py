from abc import ABC, abstractmethod
from pathlib import Path
import urllib.request
from spandrel import ImageModelDescriptor, ModelLoader
import torch


class BaseUpscaler(ABC):
    """Base class for upscaling models"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.model_descriptor = None
    
    @property
    @abstractmethod
    def name(self):
        """Model name for output labeling"""
        pass
    
    @property
    @abstractmethod
    def model_url(self):
        """URL to download model weights"""
        pass
    
    @property
    @abstractmethod
    def model_filename(self):
        """Filename for cached model"""
        pass
    
    def download_model(self):
        """Download model if not cached"""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / self.model_filename
        
        if model_path.exists():
            print(f"  {self.name}: Model cached at {model_path}")
            return str(model_path)
        
        print(f"  {self.name}: Downloading model...")
        urllib.request.urlretrieve(self.model_url, model_path)
        print(f"  {self.name}: Downloaded to {model_path}")
        
        return str(model_path)
    
    def load(self, use_fp16=False):
        """Load model onto device"""
        model_path = self.download_model()
        
        print(f"  {self.name}: Loading model...")
        self.model_descriptor = ModelLoader().load_from_file(model_path)
        assert isinstance(self.model_descriptor, ImageModelDescriptor)
        
        self.model_descriptor = self.model_descriptor.to(self.device)
        
        # Track if we're actually using FP16
        self.using_fp16 = False
        
        if use_fp16:
            try:
                self.model_descriptor = self.model_descriptor.half()
                self.using_fp16 = True
                print(f"  {self.name}: Using FP16")
            except Exception as e:
                print(f"  {self.name}: FP16 not supported, using FP32")
        
        self.model_descriptor.eval()
        print(f"  {self.name}: Ready (arch={self.model_descriptor.architecture.name}, scale={self.model_descriptor.scale}x)")
    
    @property
    def scale(self):
        """Get model's native scale factor"""
        if self.model_descriptor:
            return self.model_descriptor.scale
        return 4
    
    def __call__(self, tensor):
        """Process image tensor"""
        return self.model_descriptor(tensor)


class UltraSharpUpscaler(BaseUpscaler):
    """
    UltraSharp - ESRGAN-based upscaler
    
    Best for: General purpose, sharp details
    Strengths: Excellent edge sharpness, good on text/graphics
    Weaknesses: Can over-sharpen, may introduce slight artifacts on faces
    Speed: Fast (FP16 supported)
    """
    name = "ultrasharp"
    model_url = "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth"
    model_filename = "4x-UltraSharp.pth"


class RealESRGANUpscaler(BaseUpscaler):
    """
    Real-ESRGAN - Original ESRGAN for real-world images
    
    Best for: Photographs, natural images, balanced results
    Strengths: Smooth results, handles compression artifacts well
    Weaknesses: Less sharp than UltraSharp, can smooth fine details
    Speed: Very fast (FP16 supported, lightest model)
    """
    name = "realesrgan"
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    model_filename = "RealESRGAN_x4plus.pth"


class Nomos8kDATUpscaler(BaseUpscaler):
    """
    Nomos8k DAT - Dual Aggregation Transformer trained on high-quality photos
    
    Best for: High-quality photography, realistic textures
    Strengths: Natural texture handling, trained on premium photo dataset
    Weaknesses: Slower processing, no FP16 support
    Speed: Slow (DAT architecture, FP32 only)
    Note: Trained on Nomos8k dataset - excels at photorealistic content
    """
    name = "nomos8k"
    model_url = "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4xNomos8kDAT.pth"
    model_filename = "4xNomos8kDAT.pth"


class SwinIRUpscaler(BaseUpscaler):
    """
    SwinIR - Swin Transformer for real-world SR
    
    Best for: Complex scenes, mixed content (text + images)
    Strengths: Handles diverse degradations, good generalization
    Weaknesses: Moderate speed, no FP16 support
    Speed: Medium (Transformer architecture, FP32 only)
    Note: Uses shifted window attention mechanism
    """
    name = "swinir"
    model_url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
    model_filename = "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"

# Model registry - maps names to classes
MODEL_REGISTRY = {
    'ultrasharp': UltraSharpUpscaler,
    'realesrgan': RealESRGANUpscaler,
    'nomos8k': Nomos8kDATUpscaler,
    'swinir': SwinIRUpscaler,
}


def get_model(model_name, device='cuda'):
    """Factory function to create model instances"""
    model_class = MODEL_REGISTRY.get(model_name.lower())
    if not model_class:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return model_class(device=device)