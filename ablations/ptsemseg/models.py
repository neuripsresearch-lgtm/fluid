"""
Segmentation model architectures.
Provides factory function to get models like U-Net, DeepLab, etc.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Optional


def get_model(cfg: Dict[str, Any], n_classes: int) -> nn.Module:
    """
    Get a segmentation model based on configuration.
    
    Args:
        cfg: Configuration dict with 'arch' specifying the architecture
        n_classes: Number of output classes
    
    Returns:
        Segmentation model
    """
    arch = cfg.get('arch', 'unet').lower()
    
    if arch == 'unet':
        return UNet(n_channels=3, n_classes=n_classes, 
                   depth=cfg.get('depth', 4),
                   base_channels=cfg.get('base_channels', 64))
    
    elif arch == 'deeplabv3':
        return DeepLabV3(n_classes=n_classes, backbone='resnet50')
    
    elif arch == 'fcn':
        return FCN(n_classes=n_classes, backbone='resnet50')
    
    else:
        raise ValueError(f"Unknown architecture: {arch}")


class UNet(nn.Module):
    """
    U-Net segmentation architecture.
    Suitable for dense prediction tasks like semantic segmentation.
    """
    def __init__(self, n_channels: int = 3, n_classes: int = 11, 
                 depth: int = 4, base_channels: int = 64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth
        self.base_channels = base_channels
        
        # Encoder
        self.encoder = nn.ModuleList()
        in_ch = n_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoder.append(self._conv_block(in_ch, out_ch))
            in_ch = out_ch
        
        # Bottleneck
        self.bottleneck = self._conv_block(in_ch, in_ch * 2)
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        in_ch = in_ch * 2
        for i in range(depth - 1, -1, -1):
            out_ch = base_channels * (2 ** i)
            self.upsamples.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1))
            self.decoder.append(self._conv_block(in_ch, out_ch))
            in_ch = out_ch
        
        # Final output layer
        self.final = nn.Conv2d(base_channels, n_classes, 1)
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store encoder outputs for skip connections
        encoder_outs = []
        
        # Encoder path
        for enc_block in self.encoder:
            x = enc_block(x)
            encoder_outs.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for i, (upsample, dec_block) in enumerate(zip(self.upsamples, self.decoder)):
            x = upsample(x)
            # Skip connection
            skip = encoder_outs[-(i+1)]
            if x.shape != skip.shape:
                # Handle size mismatch (can happen with certain input sizes)
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:], 
                                                   mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)
        
        # Final output
        x = self.final(x)
        return x


class DeepLabV3(nn.Module):
    """
    DeepLabV3 segmentation architecture.
    Uses ResNet backbone with atrous spatial pyramid pooling.
    """
    def __init__(self, n_classes: int = 21, backbone: str = 'resnet50'):
        super().__init__()
        
        # Load pretrained backbone
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=True)
        elif backbone == 'resnet101':
            base_model = models.resnet101(pretrained=True)
        else:
            base_model = models.resnet50(pretrained=True)
        
        # Extract backbone features
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        
        # Simple decoder (for quick implementation)
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        out = self.decoder(features)
        # Upsample to input size
        out = torch.nn.functional.interpolate(out, size=input_shape,
                                            mode='bilinear', align_corners=False)
        return out


class FCN(nn.Module):
    """
    Fully Convolutional Network for segmentation.
    Simple baseline architecture using ResNet backbone.
    """
    def __init__(self, n_classes: int = 21, backbone: str = 'resnet50'):
        super().__init__()
        
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=True)
        elif backbone == 'resnet101':
            base_model = models.resnet101(pretrained=True)
        else:
            base_model = models.resnet50(pretrained=True)
        
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(512, n_classes, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        x = self.features(x)
        x = self.classifier(x)
        x = torch.nn.functional.interpolate(x, size=input_shape,
                                           mode='bilinear', align_corners=False)
        return x
