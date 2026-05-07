"""
Image augmentation utilities for semantic segmentation.
Provides common augmentation transforms for training.
"""

import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
from typing import Tuple, List, Optional, Callable


class Compose:
    """Compose multiple augmentations."""
    def __init__(self, transforms_list: List[Callable]):
        self.transforms = transforms_list
    
    def __call__(self, img: Image.Image, lbl: Image.Image) -> Tuple[Image.Image, Image.Image]:
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl


class RandomHorizontalFlip:
    """Randomly flip image and label horizontally."""
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, img: Image.Image, lbl: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if np.random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
        return img, lbl


class RandomVerticalFlip:
    """Randomly flip image and label vertically."""
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, img: Image.Image, lbl: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if np.random.random() < self.p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            lbl = lbl.transpose(Image.FLIP_TOP_BOTTOM)
        return img, lbl


class RandomRotation:
    """Randomly rotate image and label."""
    def __init__(self, degrees: int = 10, p: float = 0.5):
        self.degrees = degrees
        self.p = p
    
    def __call__(self, img: Image.Image, lbl: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if np.random.random() < self.p:
            angle = np.random.randint(-self.degrees, self.degrees)
            img = img.rotate(angle, resample=Image.BILINEAR)
            lbl = lbl.rotate(angle, resample=Image.NEAREST)
        return img, lbl


class RandomGaussianBlur:
    """Randomly apply Gaussian blur to image."""
    def __init__(self, radius: int = 2, p: float = 0.5):
        self.radius = radius
        self.p = p
    
    def __call__(self, img: Image.Image, lbl: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if np.random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.radius))
        return img, lbl


class RandomCrop:
    """Randomly crop image and label to given size."""
    def __init__(self, size: Tuple[int, int]):
        self.size = size
    
    def __call__(self, img: Image.Image, lbl: Image.Image) -> Tuple[Image.Image, Image.Image]:
        w, h = img.size
        th, tw = self.size
        
        if w < tw or h < th:
            return img, lbl
        
        x = np.random.randint(0, w - tw)
        y = np.random.randint(0, h - th)
        
        img = img.crop((x, y, x + tw, y + th))
        lbl = lbl.crop((x, y, x + tw, y + th))
        return img, lbl


class RandomBrightness:
    """Randomly adjust brightness of image."""
    def __init__(self, factor: float = 0.3, p: float = 0.5):
        self.factor = factor
        self.p = p
    
    def __call__(self, img: Image.Image, lbl: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if np.random.random() < self.p:
            enhancer = ImageOps.autocontrast(img)
            img = enhancer
        return img, lbl


class Identity:
    """Identity transform (no operation)."""
    def __call__(self, img: Image.Image, lbl: Image.Image) -> Tuple[Image.Image, Image.Image]:
        return img, lbl


def get_composed_augmentations(cfg: Optional[dict] = None):
    """
    Get a composition of augmentations based on config.
    
    Args:
        cfg: dict with augmentation config (e.g., {'flip_prob': 0.5, 'rotation_degrees': 10})
    
    Returns:
        Compose object with list of augmentations
    """
    if cfg is None or not cfg:
        return Compose([Identity()])
    
    augmentations = []
    
    if cfg.get('random_flip', False):
        augmentations.append(RandomHorizontalFlip(p=cfg.get('flip_prob', 0.5)))
    
    if cfg.get('random_rotation', False):
        augmentations.append(RandomRotation(degrees=cfg.get('rotation_degrees', 10),
                                           p=cfg.get('rotation_prob', 0.5)))
    
    if cfg.get('random_blur', False):
        augmentations.append(RandomGaussianBlur(radius=cfg.get('blur_radius', 2),
                                               p=cfg.get('blur_prob', 0.5)))
    
    if cfg.get('random_brightness', False):
        augmentations.append(RandomBrightness(factor=cfg.get('brightness_factor', 0.3),
                                             p=cfg.get('brightness_prob', 0.5)))
    
    if not augmentations:
        augmentations.append(Identity())
    
    return Compose(augmentations)


# Export common augmentations for 'from ptsemseg.augmentations import *'
__all__ = [
    'Compose',
    'RandomHorizontalFlip',
    'RandomVerticalFlip',
    'RandomRotation',
    'RandomGaussianBlur',
    'RandomCrop',
    'RandomBrightness',
    'Identity',
    'get_composed_augmentations'
]
