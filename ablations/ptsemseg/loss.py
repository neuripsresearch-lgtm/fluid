"""
Loss functions for semantic segmentation with hierarchical support.
Includes standard losses and tree-aware hierarchical losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union


def get_loss_function(cfg: Dict[str, Any]):
    """
    Get loss function based on configuration.
    
    Args:
        cfg: Config dict with 'name', 'params', etc.
    
    Returns:
        Loss function instance
    """
    loss_name = cfg.get('loss', {}).get('name', 'cross_entropy').lower()
    
    if loss_name == 'cross_entropy':
        return CrossEntropyLoss()
    
    elif loss_name == 'tree_loss':
        return TreeAwareLoss()
    
    elif loss_name == 'focal_loss':
        alpha = cfg.get('loss', {}).get('alpha', 0.25)
        gamma = cfg.get('loss', {}).get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_name == 'dice_loss':
        return DiceLoss()
    
    else:
        return CrossEntropyLoss()


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss for semantic segmentation."""
    
    def __init__(self, weight: Optional[torch.Tensor] = None, ignore_index: int = 250):
        super().__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, 
                root=None, use_hierarchy: bool = False) -> Union[torch.Tensor, Tuple]:
        """
        Args:
            input: (N, C, H, W) model predictions (logits)
            target: (N, H, W) ground truth labels
            root: (unused in standard CE)
            use_hierarchy: (unused in standard CE)
        
        Returns:
            loss: scalar loss value
        """
        loss = self.criterion(input, target)
        return loss


class TreeAwareLoss(nn.Module):
    """
    Hierarchical tree-aware loss for semantic segmentation.
    Penalizes mistakes less if predicted and ground-truth classes are close in hierarchy.
    """
    
    def __init__(self, ignore_index: int = 250, base_loss_weight: float = 1.0,
                 hierarchy_weight: float = 0.5):
        super().__init__()
        self.ignore_index = ignore_index
        self.base_loss_weight = base_loss_weight
        self.hierarchy_weight = hierarchy_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    
    def forward(self, input: torch.Tensor, target: torch.Tensor,
                root=None, use_hierarchy: bool = True) -> Union[torch.Tensor, Tuple]:
        """
        Args:
            input: (N, C, H, W) model predictions
            target: (N, H, W) ground truth labels
            root: Tree root node (for hierarchy awareness)
            use_hierarchy: Whether to use hierarchical weighting
        
        Returns:
            loss: scalar loss or tuple (main_loss, auxiliary_loss) if use_hierarchy=True
        """
        # Base cross-entropy loss
        ce_loss = self.ce_loss(input, target)
        
        # Mask out ignore_index
        mask = target != self.ignore_index
        ce_loss = ce_loss * mask.float()
        
        if not use_hierarchy or root is None:
            # Fall back to standard CE loss
            return ce_loss.mean()
        
        # Hierarchical weighting: reduce penalty for hierarchically close mistakes
        # This is a simplified version - can be extended with actual tree distances
        B, C, H, W = input.shape
        
        # Get predictions
        preds = input.argmax(dim=1)  # (B, H, W)
        
        # Simple hierarchy weighting: closer in class index = smaller penalty
        # In a full implementation, this would use tree distances
        class_distance = torch.abs(preds.float() - target.float())
        hierarchy_weight = torch.exp(-class_distance / (C + 1))
        
        weighted_loss = ce_loss * hierarchy_weight * mask.float()
        
        main_loss = weighted_loss.mean()
        aux_loss = ce_loss.mean() - main_loss  # Auxiliary loss for monitoring
        
        return (main_loss, aux_loss)


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    From: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 ignore_index: int = 250):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, input: torch.Tensor, target: torch.Tensor,
                root=None, use_hierarchy: bool = False):
        """
        Args:
            input: (N, C, H, W) logits
            target: (N, H, W) labels
        """
        p = F.softmax(input, dim=1)
        ce_loss = F.cross_entropy(input, target, reduction='none', 
                                 ignore_index=self.ignore_index)
        
        # Get probability of true class
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = p_t.masked_fill(target == self.ignore_index, 1.0)
        
        # Focal loss
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """
    Dice loss (F1-based loss) for segmentation.
    Works well for imbalanced datasets.
    """
    
    def __init__(self, ignore_index: int = 250, smooth: float = 1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
    
    def forward(self, input: torch.Tensor, target: torch.Tensor,
                root=None, use_hierarchy: bool = False):
        """
        Args:
            input: (N, C, H, W) logits
            target: (N, H, W) labels
        """
        p = F.softmax(input, dim=1)
        
        # Create one-hot target
        n_classes = input.shape[1]
        target_one_hot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
        
        # Mask ignore_index
        mask = (target != self.ignore_index).float()
        
        # Compute Dice score per class
        intersection = (p * target_one_hot * mask.unsqueeze(1)).sum()
        cardinality = (p.sum() + target_one_hot.sum()) * mask.unsqueeze(1).sum()
        
        dice_score = 2.0 * intersection / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score
        
        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function (CE + Dice or similar).
    """
    
    def __init__(self, ignore_index: int = 250, ce_weight: float = 0.5,
                 dice_weight: float = 0.5):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        self.ce_loss = CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor,
                root=None, use_hierarchy: bool = False):
        ce = self.ce_loss(input, target, root=root, use_hierarchy=use_hierarchy)
        dice = self.dice_loss(input, target, root=root, use_hierarchy=use_hierarchy)
        
        combined = self.ce_weight * ce + self.dice_weight * dice
        return combined
