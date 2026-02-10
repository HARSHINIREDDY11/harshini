import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def save_activation(module, input, output):
            self.activations = output

        # For Swin Transformer, target_layer might be the last stage
        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_full_backward_hook(save_gradient)

    def generate_heatmap(self, input_tensor, target_class=None):
        # input_tensor shape: [1, V, 3, 224, 224]
        # We need to run backward pass to get gradients
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        # Swin feature maps are typically [B*V, C, H_f, W_f] if hooked inside the backbone
        # In our model, we flatten B and V before backbone
        # self.activations shape: [B*V, C, H_f, W_f]
        grads = self.gradients
        acts = self.activations
        
        # Calculate weights per channel
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize and upscale to 224x224
        heatmaps = []
        for i in range(cam.shape[0]):
            c = cam[i, 0].detach().cpu().numpy()
            c = (c - c.min()) / (c.max() - c.min() + 1e-8)
            c = cv2.resize(c, (224, 224))
            heatmaps.append(c)
            
        return heatmaps

def apply_heatmap(img, heatmap, intensity=0.5):
    # img is 224x224 RGB
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    result = cv2.addWeighted(np.uint8(img), 1 - intensity, heatmap_color, intensity, 0)
    return result
