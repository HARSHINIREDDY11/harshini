import os
import random
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TrashNetMultiViewDataset(Dataset):
    def __init__(self, root_dir, num_views=(2, 4), transform=None, split='train', val_split=0.2):
        self.root_dir = root_dir
        # Only list actual directories and skip hidden ones like .DS_Store
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.num_views_range = num_views
        self.transform = transform
        self.split = split
        self.val_split = val_split

        # Collect all image paths per class
        self.samples_by_class = {}
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            self.samples_by_class[cls] = [os.path.join(cls_path, img) for img in os.listdir(cls_path)]

        # Create a list of all images
        all_images = []
        for cls, imgs in self.samples_by_class.items():
            for img in imgs:
                all_images.append((img, cls))

        # Split into train/val
        random.seed(42)  # For reproducibility
        random.shuffle(all_images)
        split_idx = int(len(all_images) * (1 - self.val_split))
        if self.split == 'train':
            self.all_images = all_images[:split_idx]
        elif self.split == 'val':
            self.all_images = all_images[split_idx:]
        else:
            raise ValueError("Split must be 'train' or 'val'")

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        anchor_img_path, cls = self.all_images[idx]
        
        # Determine number of views for this sample
        n_views = random.randint(self.num_views_range[0], self.num_views_range[1])
        
        # Pick n_views-1 additional images from the same class
        other_imgs = random.sample(self.samples_by_class[cls], k=min(n_views - 1, len(self.samples_by_class[cls]) - 1))
        view_paths = [anchor_img_path] + other_imgs
        
        # Ensure we always have the same number of views for batching? 
        # Actually, the requirement says "2-4". For modern CNNs/Transformers, we can pad with zeros or just use a fixed number for simplicity.
        # Let's fix it to 4 views for consistency in the 3D CNN, padding with copies if necessary.
        target_views = 4
        while len(view_paths) < target_views:
            view_paths.append(random.choice(view_paths))
            
        images = []
        for path in view_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            
        # Apply Shared Augmentations (Stage 3)
        if self.transform:
            # Albumentations can apply same transform to multiple images
            # But we need to use 'additional_targets'
            # Or just set a seed
            seed = random.randint(0, 1000000)
            transformed_images = []
            for img in images:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                t = self.transform(image=img)
                transformed_images.append(t['image'])
            images = transformed_images
        
        # Stack views along a new dimension: [V, C, H, W]
        stacked_images = torch.stack(images)
        
        label = self.class_to_idx[cls]
        
        return stacked_images, label


def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

if __name__ == "__main__":
    # Test logic
    from data_setup import setup_directories, generate_mock_data
    setup_directories(".")
    generate_mock_data("./trashnet")
    
    dataset = TrashNetMultiViewDataset("./trashnet", transform=get_transforms(True))
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    images, labels = next(iter(loader))
    print(f"Batch shape: {images.shape}") # Should be [B, V, C, H, W] -> [4, 4, 3, 224, 224]
    print(f"Labels: {labels}")
