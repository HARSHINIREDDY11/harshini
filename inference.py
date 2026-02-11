import torch
import time
import numpy as np
import cv2
from model import MultiViewSwin3DCNN

def measure_fps(model, device, num_views=4, input_size=(224, 224), iterations=100):
    model.eval()
    dummy_input = torch.randn(1, num_views, 3, *input_size).to(device)
    
    # Warm up
    for _ in range(10):
        _ = model(dummy_input)
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            
    end_time = time.time()
    total_time = end_time - start_time
    fps = iterations / total_time
    print(f"Average Inference Speed: {fps:.2f} FPS")
    print(f"Total time for {iterations} iterations: {total_time:.4f} seconds")
    return fps

def real_time_simulation(model, device, classes):
    """Simulates a real-time multi-view capture and inference."""
    print("\nSimulating real-time multi-view inference...")
    # Simulate 4 camera views
    views = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(4)]
    
    # Preprocess
    input_tensor = []
    for img in views:
        img = img.astype(np.float32) / 255.0
        # Normalization
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))
        input_tensor.append(img)
    
    input_tensor = torch.from_numpy(np.array([input_tensor])).float().to(device)
    
    model.eval()
    with torch.no_grad():
        start = time.time()
        logits = model(input_tensor)
        latency = (time.time() - start) * 1000
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)
        
    print(f"Prediction: {classes[pred.item()]} ({conf.item()*100:.2f}%)")
    print(f"Latency: {latency:.2f} ms")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MultiViewSwin3DCNN(num_classes=6).to(device)
    
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    measure_fps(model, device)
    real_time_simulation(model, device, classes)
