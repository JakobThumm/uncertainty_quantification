import torch
import torch.nn as nn
from timm import create_model
# Customize for having freedom on number of heads 
class ViTPoseFull(nn.Module):
    def __init__(
        self,
        img_size=(256,192),
        patch_size=16,
        num_keypoints=13,
        vit_model='vit_base_patch16_224',
        pretrained=False,
        embed_dim=None,
        depth=None,
        num_heads=None,
        mlp_ratio=None
    ):
        super().__init__()
        
        self.img_size   = img_size
        self.patch_size = patch_size

        # How many patches per dimension
        self.grid_h = img_size[0] // patch_size
        self.grid_w = img_size[1] // patch_size

        # Build the ViT backbone, overriding patch_size
        create_kwargs = {
            'pretrained': pretrained,
            'img_size': img_size,
            'patch_size': patch_size,
            'num_classes': 0,          # strip off classification head
        }
        if embed_dim   is not None: create_kwargs['embed_dim']   = embed_dim
        if depth       is not None: create_kwargs['depth']       = depth
        if num_heads   is not None: create_kwargs['num_heads']   = num_heads
        if mlp_ratio   is not None: create_kwargs['mlp_ratio']   = mlp_ratio

        self.backbone = create_model(vit_model, **create_kwargs)
        self.embed_dim = self.backbone.embed_dim  # whatever ended up being used
        
        # Optional 1×1 conv to refine the feature maps
        self.reshape_conv = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1)

        # Upsampling head → num_keypoints heatmaps
        m1, m2, m3 = self.embed_dim//2, self.embed_dim//4, self.embed_dim//8
        self.head = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, m1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(m1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(m1, m2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(m2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(m2, m3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(m3), nn.ReLU(inplace=True),
            nn.Conv2d(m3, num_keypoints, kernel_size=1)
        )

    def forward(self, x):
        B = x.shape[0]
        # → (B, 1 + N, C)
        feats = self.backbone.forward_features(x)

        # drop the CLS token, → (B, N, C)
        feats = feats[:, 1:, :]

        # rearrange to (B, C, grid_h, grid_w)
        feats = feats.permute(0,2,1).contiguous()
        feats = feats.view(B, self.embed_dim, self.grid_h, self.grid_w)

        feats = self.reshape_conv(feats)
        out = self.head(feats)  # (B, num_keypoints, Hout, Wout)
        return out
