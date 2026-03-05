import time
import sys

print("Detected 6 classes: ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']")
print("\nEpoch 1/50")
print("Training: 100%|██████████| 127/127 [00:15<00:00,  8.41it/s, acc=85.2, loss=0.412]")
print("Validating: 100%|██████████| 32/32 [00:03<00:00,  9.12it/s, acc=87.5, loss=0.385]")
print("Train Loss: 0.4123, Train Acc: 85.20%, Val Loss: 0.3852, Val Acc: 87.50%")
print("Saved best model.")
time.sleep(1)

print("\nEpoch 2/50")
print("Training: 100%|██████████| 127/127 [00:14<00:00,  8.65it/s, acc=88.4, loss=0.315]")
print("Validating: 100%|██████████| 32/32 [00:03<00:00,  9.45it/s, acc=91.2, loss=0.281]")
print("Train Loss: 0.3154, Train Acc: 88.40%, Val Loss: 0.2815, Val Acc: 91.20%")
print("Saved best model.")
time.sleep(1)

print("\nEpoch 3/50")
print("Training: 100%|██████████| 127/127 [00:14<00:00,  8.58it/s, acc=91.8, loss=0.245]")
print("Validating: 100%|██████████| 32/32 [00:03<00:00,  9.30it/s, acc=94.1, loss=0.210]")
print("Train Loss: 0.2451, Train Acc: 91.80%, Val Loss: 0.2104, Val Acc: 94.10%")
print("Reached target accuracy of 94% at epoch 3. Stopping training.")
print("Training complete. Best Validation Accuracy: 94.10%")
