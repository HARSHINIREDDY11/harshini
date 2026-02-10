import os
import shutil
import random
import cv2
import numpy as np

def setup_directories(base_path):
    """Creates the TrashNet directory structure."""
    classes = [
        'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'
    ]
    dataset_path = os.path.join(base_path, 'trashnet')
    for cls in classes:
        os.makedirs(os.path.join(dataset_path, cls), exist_ok=True)
    return dataset_path

def generate_mock_data(dataset_path, num_samples=20):
    """Generates synthetic mock images for development purposes."""
    classes = [
        'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'
    ]
    
    # Distinct colors for easy learning (BGR format for OpenCV)
    class_colors = {
        'cardboard': (40, 60, 100),   # Brownish
        'glass': (255, 255, 200),     # Cyan/Light Blue
        'metal': (192, 192, 192),     # Gray
        'paper': (255, 255, 255),     # White
        'plastic': (0, 0, 255),       # Red
        'trash': (30, 30, 30)         # Dark Gray/Black
    }
    # Check if dataset is already populated (e.g. real data)
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        if os.path.exists(cls_path) and len(os.listdir(cls_path)) > 0:
            print(f"Directory {cls_path} not empty. Skipping mock data generation for this class.")
            continue
            
    print(f"Generating {num_samples} mock images per class in {dataset_path}...")
    
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        for i in range(num_samples):
            # Create a dummy colored image
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Base color + random noise
            base_color = class_colors[cls]
            noise = np.random.randint(-20, 20, (224, 224, 3))
            img_data = np.full((224, 224, 3), base_color, dtype=np.int16) + noise
            img_data = np.clip(img_data, 0, 255).astype(np.uint8)
            img[:] = img_data
            
            # Add some text to distinguish
            cv2.putText(img, f"{cls}_{i}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            img_name = f"{cls}{i+1}.jpg"
            cv2.imwrite(os.path.join(cls_path, img_name), img)
            
    # Mock segmentation masks (simple circle in the middle)
    mask_path = os.path.join(os.path.dirname(dataset_path), 'masks')
    os.makedirs(mask_path, exist_ok=True)
    for cls in classes:
        cls_mask_path = os.path.join(mask_path, cls)
        os.makedirs(cls_mask_path, exist_ok=True)
        for i in range(num_samples):
            mask = np.zeros((224, 224), dtype=np.uint8)
            cv2.circle(mask, (112, 112), 50, 255, -1)
            cv2.imwrite(os.path.join(cls_mask_path, f"{cls}{i+1}_mask.jpg"), mask)

if __name__ == "__main__":
    base_dir = "."
    ds_path = setup_directories(base_dir)
    generate_mock_data(ds_path)
    print("Setup complete.")
