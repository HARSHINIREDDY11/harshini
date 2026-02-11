import torch
import torch.nn as nn
import torchvision.models as models
import timm

class CNNBaseline(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(CNNBaseline, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        # Expects [B, 3, 224, 224]
        return self.resnet(x)

class SwinSingleView(nn.Module):
    def __init__(self, num_classes=6, model_name='swin_tiny_patch4_window7_224', pretrained=True):
        super(SwinSingleView, self).__init__()
        self.swin = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        # Expects [B, 3, 224, 224]
        return self.swin(x)
