# Image Upscaler - GPU-Accelerated AI Image Enhancement

A professional GPU-accelerated image upscaling and enhancement tool using state-of-the-art AI models. Built as a hands-on learning project for GPU computing, PyTorch, and AI inference fundamentals.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)

## Features

- 🚀 **GPU-Accelerated**: Leverages CUDA for fast processing on NVIDIA GPUs
- 🎨 **Multiple AI Models**: Choose from 4 different architectures optimized for different content types
- ⚡ **FP16 Support**: 2-3x faster processing with half-precision on compatible models
- 🧩 **Smart Tiling**: Automatically handles large images that exceed VRAM limits
- 🎯 **Enhance-Only Mode**: Improve quality without changing image dimensions
- 💻 **Dual Interface**: Interactive CLI for beginners, direct command-line for power users
- 📊 **A/B Testing**: Process with multiple models simultaneously for quality comparison

## Models

| Model | Architecture | Best For | Speed | FP16 |
|-------|-------------|----------|-------|------|
| **UltraSharp** | ESRGAN | Sharp details, text, graphics | Fast | ✅ |
| **Real-ESRGAN** | ESRGAN | Balanced, compression artifacts | Fastest | ✅ |
| **Nomos8k** | DAT | High-quality photos, realistic textures | Slow | ❌ |
| **SwinIR** | Transformer | Complex scenes, mixed content | Medium | ❌ |

## Requirements

- **OS**: Linux (tested on Ubuntu 22.04 via WSL2)
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4090, 24GB VRAM)
- **Python**: 3.10+
- **CUDA**: 12.0+

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/kevinmcmahonxyz/image-upscaler.git
cd image-upscaler
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify GPU access
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

### 5. Set up CLI (optional, for system-wide access)
```bash
./setup-cli.sh
source ~/.bashrc
```

## Usage

### Interactive Mode (Recommended for Beginners)
```bash
img-enhance
```

Follow the step-by-step prompts to select your image, model, and settings.

### Direct Command-Line Mode
```bash
# Basic upscaling
img-enhance -i photo.jpg

# Use specific model with FP16
img-enhance -i photo.jpg -m nomos8k --fp16

# Enhance quality without changing size
img-enhance -i blurry.jpg --enhance-only

# Compare multiple models
img-enhance -i photo.jpg -m ultrasharp,realesrgan,nomos8k --fp16

# 4x upscale with all models
img-enhance -i small.jpg -s 4 -m all
```

### Common Use Cases

**Small web image → High resolution:**
```bash
img-enhance -i screenshot.png -s 4 -m ultrasharp --fp16
# Input: 500x300 → Output: 2000x1200 (sharp, clean)
```

**Blurry photo → Enhanced quality:**
```bash
img-enhance -i blurry.jpg -m nomos8k --enhance-only --fp16
# Input: 2000x1500 → Output: 2000x1500 (deblurred, denoised)
```

**Compare models for best quality:**
```bash
img-enhance -i photo.jpg -m all --fp16
# Outputs: photo_ultrasharp_x4.jpg, photo_realesrgan_x4.jpg, etc.
```

## Command-Line Options
```bash
img-enhance --help
```

**Key Options:**
- `-i, --input PATH` - Input image file (required)
- `-o, --output PATH` - Output path (default: auto-generated)
- `-s, --scale N` - Scale factor: 2 or 4 (default: auto-detect)
- `-m, --models MODEL` - Model(s) to use (default: ultrasharp)
- `--fp16` - Use half-precision for faster processing
- `--enhance-only` - Enhance without changing size
- `--interactive, -I` - Launch interactive mode

## Project Structure
```
image-upscaler/
├── img-enhance          # CLI entry point (executable)
├── upscale.py          # Core upscaling logic
├── interactive.py      # Interactive mode interface
├── models.py           # Model abstraction & registry
├── utils.py            # VRAM estimation & tiling utilities
├── setup-cli.sh        # CLI setup script
├── requirements.txt    # Python dependencies
├── test_images/        # Test images
└── models/             # Downloaded model weights (auto-created)
```

## Performance

Tested on RTX 4090 (24GB VRAM):

| Image Size | Model | FP16 | Time |
|-----------|-------|------|------|
| 500×300 | UltraSharp | Yes | 0.28s |
| 500×300 | RealESRGAN | Yes | 0.08s |
| 500×300 | Nomos8k | No | 3.46s |
| 800×600 | UltraSharp | Yes | 0.89s |
| 2000×1500 (enhance) | Nomos8k | No | ~5s |

## Learning Goals Achieved

This project was built as a hands-on learning experience for:

✅ **GPU Computing Fundamentals**
- CUDA concepts: parallel execution, memory hierarchy
- Understanding when/why GPU acceleration matters
- Memory constraints and optimization strategies

✅ **PyTorch/CUDA Integration**
- CPU ↔ GPU data movement
- Tensor operations on GPU
- VRAM management and estimation
- Mixed precision (FP16) inference

✅ **ML Inference Pipeline**
- Model loading and initialization
- Image preprocessing/postprocessing
- Batching and tiling strategies
- Performance optimization techniques

✅ **Software Engineering**
- Clean architecture with abstractions
- CLI design and user experience
- Git workflows and documentation
- Practical problem-solving with AI models

## Troubleshooting

**CUDA out of memory error:**
- The tool automatically tiles large images, but if you still get errors:
- Try `--fp16` flag (reduces VRAM by ~50%)
- Use a smaller scale factor: `-s 2` instead of `-s 4`
- Process one model at a time instead of `all`

**Slow processing:**
- Use `--fp16` for 2-3x speedup (compatible with ultrasharp, realesrgan)
- Choose faster models: realesrgan > ultrasharp > swinir > nomos8k
- For large images, expect tiling which is slower but necessary

**Model download fails:**
- Check internet connection
- Models download automatically on first use (~50-300MB each)
- Downloaded models are cached in `models/` directory

## Contributing

This is a learning project, but suggestions and improvements are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests with improvements
- Share your results and use cases

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **Real-ESRGAN**: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **UltraSharp**: [Kim2091/UltraSharp](https://huggingface.co/Kim2091/UltraSharp)
- **SwinIR**: [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)
- **Nomos8k DAT**: [Phips/4xNomos8kDAT](https://huggingface.co/Phips/4xNomos8kDAT)
- **Spandrel**: [chaiNNer-org/spandrel](https://github.com/chaiNNer-org/spandrel) - Universal model loader

## Author

Built by Kevin McMahon ([@kevinmcmahonxyz](https://github.com/kevinmcmahonxyz)) as a hands-on learning project for GPU computing and AI inference.

---

**⭐ Star this repo if you find it useful!**
