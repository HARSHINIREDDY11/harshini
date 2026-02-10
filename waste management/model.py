import torch
import torch.nn as nn
import timm

class MultiViewResNet3DCNN(nn.Module):
    def __init__(self, num_classes=10, model_name='resnet18', pretrained=True):
        super(MultiViewResNet3DCNN, self).__init__()

        # ResNet Backbone (for compatibility with existing weights)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=(3,))

        # Get feature dimensions (for resnet18, stage 3 output is 256 channels, 14x14 resolution)
        dummy_in = torch.randn(1, 3, 224, 224)
        dummy_out = self.backbone(dummy_in)[0]  # List of features, take last
        self.feat_dim = dummy_out.shape[1]  # Channels (256 for ResNet18)
        self.spatial_dim = dummy_out.shape[2]  # H/W (14 for ResNet18)

        # Multi-View Fusion - 3D CNN (optimized for speed)
        self.fusion_3d = nn.Sequential(
            nn.Conv3d(self.feat_dim, self.feat_dim // 4, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(self.feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # Reduce to 1D vector
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim // 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # Optional Segmentation Head (simplified)
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(self.feat_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # [B, 1, 224, 224]
            nn.Sigmoid()
        )

    def forward(self, x, return_seg=False):
        # x shape: [B, V, 3, 224, 224]
        batch_size, num_views, C, H, W = x.shape

        # Flatten B and V to process all views through ResNet (Shared Weights)
        x = x.view(batch_size * num_views, C, H, W)

        # Extract features (list form from features_only=True)
        feat_list = self.backbone(x)
        features = feat_list[-1] # [B*V, C_f, H_f, W_f]

        # Reshape for 3D CNN: [B, C_f, V, H_f, W_f]
        _, C_f, H_f, W_f = features.shape
        features = features.view(batch_size, num_views, C_f, H_f, W_f)
        features = features.transpose(1, 2) # [B, C_f, V, H_f, W_f]

        # Fuse views
        fused = self.fusion_3d(features) # [B, C_f//4, 1, 1, 1]
        fused = fused.view(batch_size, -1)

        # Classification
        logits = self.classifier(fused)

        if return_seg:
            # For segmentation, we might want to use the fused features or per-view features
            # Let's use the first view's features as a simplified proxy or the average feature
            # Real segmentation on multi-view is complex, but here's a proof-of-concept
            avg_feat = features.mean(dim=2) # Average across views [B, C_f, H_f, W_f]
            # Upsample to 224x224
            mask = self.seg_head(avg_feat)
            # Adjust mask size to exactly 224x224 if needed (ResNet at Stage 3 is 14x14)
            # 14 -> 28 -> 56 -> 112 -> 224 (4 steps of 2x)
            return logits, mask

        return logits

class MultiViewSwin3DCNN(nn.Module):
    def __init__(self, num_classes=10, model_name='swin_tiny_patch4_window7_224', pretrained=True):
        super(MultiViewSwin3DCNN, self).__init__()

        # Swin Transformer Backbone (for high accuracy)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=(3,))

        # Get feature dimensions (for resnet18, stage 3 output is 256 channels, 14x14 resolution)
        dummy_in = torch.randn(1, 3, 224, 224)
        dummy_out = self.backbone(dummy_in)[0]  # List of features, take last
        self.feat_dim = dummy_out.shape[1]  # Channels (256 for ResNet18)
        self.spatial_dim = dummy_out.shape[2]  # H/W (14 for ResNet18)

        # Multi-View Fusion - 3D CNN (optimized for speed)
        self.fusion_3d = nn.Sequential(
            nn.Conv3d(self.feat_dim, self.feat_dim // 4, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(self.feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # Reduce to 1D vector
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim // 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # Optional Segmentation Head (simplified)
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(self.feat_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # [B, 1, 224, 224]
            nn.Sigmoid()
        )

    def forward(self, x, return_seg=False):
        # x shape: [B, V, 3, 224, 224]
        batch_size, num_views, C, H, W = x.shape
        
        # Flatten B and V to process all views through Swin (Shared Weights)
        x = x.view(batch_size * num_views, C, H, W)
        
        # Extract features (list form from features_only=True)
        feat_list = self.backbone(x)
        features = feat_list[-1] # [B*V, C_f, H_f, W_f]
        
        # Reshape for 3D CNN: [B, C_f, V, H_f, W_f]
        _, C_f, H_f, W_f = features.shape
        features = features.view(batch_size, num_views, C_f, H_f, W_f)
        features = features.transpose(1, 2) # [B, C_f, V, H_f, W_f]
        
        # Fuse views
        fused = self.fusion_3d(features) # [B, C_f//4, 1, 1, 1]
        fused = fused.view(batch_size, -1)
        
        # Classification
        logits = self.classifier(fused)
        
        if return_seg:
            # For segmentation, we might want to use the fused features or per-view features
            # Let's use the first view's features as a simplified proxy or the average feature
            # Real segmentation on multi-view is complex, but here's a proof-of-concept
            avg_feat = features.mean(dim=2) # Average across views [B, C_f, H_f, W_f]
            # Upsample to 224x224
            mask = self.seg_head(avg_feat)
            # Adjust mask size to exactly 224x224 if needed (Swin-Tiny at Stage 3 is 7x7)
            # 7 -> 14 -> 28 -> 56 -> 112 -> 224 (5 steps of 2x)
            return logits, mask
            
        return logits

if __name__ == "__main__":
    model = MultiViewSwin3DCNN(num_classes=10)
    dummy_input = torch.randn(2, 4, 3, 224, 224) # 2 batches, 4 views
    output = model(dummy_input)
    print(f"Classification Output Shape: {output.shape}")

    output_seg, mask = model(dummy_input, return_seg=True)
    print(f"Segmentation Mask Shape: {mask.shape}")
