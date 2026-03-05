import torch
import torch.nn as nn
import time
from model import MultiViewSwin3DCNN

def quantize_model(model_path="best_model.pth", output_path="quantized_model.pth"):
    print(f"Loading model for quantization: {model_path}")
    device = torch.device("cpu") # Quantization is primary for CPU
    
    model = MultiViewSwin3DCNN(num_classes=6)
    if torch.os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    
    print("Applying Dynamic Quantization...")
    # Quantize Linear and Conv layers if supported (Linear is most common for dynamic)
    # Swin uses many Linear layers for attention
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear, nn.Conv3d}, 
        dtype=torch.qint8
    )
    
    print(f"Saving quantized model to {output_path}")
    torch.save(quantized_model.state_dict(), output_path)
    return quantized_model

def benchmark(model, name="Model"):
    dummy_input = torch.randn(1, 4, 3, 224, 224)
    model.eval()
    
    # Warmup
    for _ in range(5):
        _ = model(dummy_input)
        
    start = time.time()
    iterations = 50
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    end = time.time()
    
    fps = iterations / (end - start)
    print(f"{name} Speed: {fps:.2f} FPS")
    return fps

if __name__ == "__main__":
    orig_model = MultiViewSwin3DCNN(num_classes=6)
    bench_orig = benchmark(orig_model, "Original Model")
    
    q_model = quantize_model()
    bench_q = benchmark(q_model, "Quantized Model")
    
    improvement = (bench_q / bench_orig - 1) * 100
    print(f"Speed Improvement: {improvement:.2f}%")
