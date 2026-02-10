import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.amp import autocast, GradScaler
from dataset import TrashNetMultiViewDataset, get_transforms
from model import MultiViewSwin3DCNN

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits shape [B, 1, H, W], targets shape [B, 1, H, W]
        probs = logits # Already sigmoid in model.py
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice

def train_one_epoch(model, loader, optimizer, criterion_cls, criterion_seg, device, grad_clip=1.0, use_seg=False, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type):
            if use_seg:
                # We need masks. For now, since it's synthetic multi-view,
                # we don't have real masks unless we load them.
                # I'll simulate a mask target if not provided.
                logits, masks = model(images, return_seg=True)
                # Dummy target mask (circle in middle)
                target_masks = torch.zeros_like(masks).to(device)
                # ... logic to create better dummy masks could go here ...

                loss_cls = criterion_cls(logits, labels)
                loss_seg = criterion_seg(masks, target_masks)
                loss = loss_cls + 0.5 * loss_seg
            else:
                logits = model(images)
                loss = criterion_cls(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=total_loss/(total/loader.batch_size), acc=100.*correct/total)

    return total_loss / len(loader), 100. * correct / total

def validate_one_epoch(model, loader, criterion_cls, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion_cls(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=total_loss/(total/loader.batch_size), acc=100.*correct/total)

    return total_loss / len(loader), 100. * correct / total

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Multi-View Waste Segmentation Model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    weight_decay = 0.05
    patience = 10  # Early stopping patience

    # Dataset
    data_dir = "./trashnet"
    if not os.path.exists(data_dir):
        print("Dataset not found! Generating mock data...")
        from data_setup import setup_directories, generate_mock_data
        setup_directories(".")
        generate_mock_data(data_dir)

    from dataset import TrashNetMultiViewDataset, get_transforms
    train_ds = TrashNetMultiViewDataset(data_dir, transform=get_transforms(is_train=True), split='train')
    val_ds = TrashNetMultiViewDataset(data_dir, transform=get_transforms(is_train=False), split='val')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    # Dynamically check classes
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    num_classes = len(classes)
    print(f"Detected {num_classes} classes: {classes}")

    model = MultiViewSwin3DCNN(num_classes=num_classes).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = DiceLoss()

    # Mixed precision scaler
    scaler = GradScaler() if device.type == 'cuda' else None

    best_val_acc = 0
    patience_counter = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion_cls, criterion_seg, device, scaler=scaler)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion_cls, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best Val Acc: {best_val_acc:.2f}%")
                break

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
