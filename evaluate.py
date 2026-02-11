import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

from dataset import TrashNetMultiViewDataset, get_transforms
from model import MultiViewSwin3DCNN
from baselines import CNNBaseline, SwinSingleView

def evaluate_model(model, loader, device, model_type='multi-view'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Evaluating {model.__class__.__name__}"):
            images, labels = images.to(device), labels.to(device)

            if model_type == 'single-view':
                # Take only the first view [B, V, C, H, W] -> [B, C, H, W]
                images = images[:, 0, :, :, :]

            logits = model(images)
            _, preds = logits.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./trashnet"
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    # Dataset
    test_ds = TrashNetMultiViewDataset(data_dir, transform=get_transforms(is_train=False))
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    # Models to benchmark (Stage 9)
    models_to_test = [
        ("CNN Baseline", CNNBaseline(num_classes=6).to(device), 'single-view'),
        ("Swin Single-View", SwinSingleView(num_classes=6).to(device), 'single-view'),
        ("Swin + 3D CNN (Multi-View)", MultiViewSwin3DCNN(num_classes=6).to(device), 'multi-view')
    ]

    # results storage
    results = {}

    for name, model, m_type in models_to_test:
        print(f"\n--- Benchmarking: {name} ---")
        # In a real scenario, you would load pre-trained weights here
        # torch.load(...)

        preds, labels = evaluate_model(model, test_loader, device, model_type=m_type)
        # Manual accuracy calculation
        correct = sum(1 for p, l in zip(preds, labels) if p == l)
        acc = correct / len(labels)
        # Use real accuracy, no hardcoding
        results[name] = acc

        print(f"Accuracy: {acc:.4f}")

        # Manual classification report
        print("Classification Report:")
        print("Class\t\tPrecision\tRecall\t\tF1-Score\tSupport")
        for i, cls in enumerate(classes):
            tp = sum(1 for p, l in zip(preds, labels) if p == i and l == i)
            fp = sum(1 for p, l in zip(preds, labels) if p == i and l != i)
            fn = sum(1 for p, l in zip(preds, labels) if p != i and l == i)
            tn = sum(1 for p, l in zip(preds, labels) if p != i and l != i)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = sum(1 for l in labels if l == i)

            print(f"{cls}\t\t{precision:.4f}\t\t{recall:.4f}\t\t{f1:.4f}\t\t{support}")

        # Class-wise accuracy
        for i, cls in enumerate(classes):
            cls_labels = [l for l in labels if l == i]
            cls_preds = [p for p, l in zip(preds, labels) if l == i]
            if cls_labels:
                cls_correct = sum(1 for p, l in zip(cls_preds, cls_labels) if p == l)
                cls_acc = cls_correct / len(cls_labels)
                print(f"{cls} accuracy: {cls_acc:.4f}")



    print("\n--- Final Benchmark Results ---")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")

if __name__ == "__main__":
    main()
