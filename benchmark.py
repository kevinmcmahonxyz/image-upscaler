import subprocess
import time

test_images = [
    ("test_images/small_test.jpg", "Small (500x300)"),
    ("test_images/medium_test.jpg", "Medium (800x600)"),
    ("test_images/large_test.jpg", "Large (2000x1500)")
]

print("=== Performance Benchmark: FP32 vs FP16 ===\n")

for img_path, description in test_images:
    print(f"Testing {description}:")
    
    # FP32 test
    start = time.time()
    subprocess.run([
        "python", "upscale.py",
        "--input", img_path,
        "--output", "outputs/bench_fp32.png",
        "--model", "UltraSharp"
    ], capture_output=True)
    fp32_time = time.time() - start
    
    # FP16 test
    start = time.time()
    subprocess.run([
        "python", "upscale.py",
        "--input", img_path,
        "--output", "outputs/bench_fp16.png",
        "--model", "UltraSharp",
        "--fp16"
    ], capture_output=True)
    fp16_time = time.time() - start
    
    speedup = fp32_time / fp16_time
    print(f"  FP32: {fp32_time:.2f}s")
    print(f"  FP16: {fp16_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x\n")