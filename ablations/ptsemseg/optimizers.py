"""
Optimizer utilities for semantic segmentation training.
Provides factory functions for common optimizers.
"""

import torch
from torch import optim
from typing import Dict, Any, Optional


def get_optimizer(cfg: Dict[str, Any], model) -> torch.optim.Optimizer:
    """
    Get optimizer based on config.
    
    Args:
        cfg: Configuration dict with 'name', 'lr', 'weight_decay', etc.
        model: PyTorch model to optimize
    
    Returns:
        Configured optimizer instance
    """
    opt_name = cfg.get('optimizer', 'adam').lower()
    lr = cfg.get('lr', 1e-4)
    weight_decay = cfg.get('weight_decay', 0.0)
    
    if opt_name == 'adam':
        beta1 = cfg.get('beta1', 0.9)
        beta2 = cfg.get('beta2', 0.999)
        return optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), 
                         weight_decay=weight_decay)
    
    elif opt_name == 'adamw':
        beta1 = cfg.get('beta1', 0.9)
        beta2 = cfg.get('beta2', 0.999)
        return optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2),
                          weight_decay=weight_decay)
    
    elif opt_name == 'sgd':
        momentum = cfg.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                        weight_decay=weight_decay)
    
    elif opt_name == 'rmsprop':
        alpha = cfg.get('alpha', 0.99)
        return optim.RMSprop(model.parameters(), lr=lr, alpha=alpha,
                            weight_decay=weight_decay)
    
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
    """
    Get learning rate scheduler based on config.
    
    Args:
        cfg: Configuration dict with 'name', 'step_size', 'gamma', etc.
        optimizer: PyTorch optimizer
    
    Returns:
        Configured scheduler instance or None
    """
    scheduler_name = cfg.get('scheduler', None)
    
    if scheduler_name is None:
        return None
    
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'steplr':
        step_size = cfg.get('step_size', 10)
        gamma = cfg.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_name == 'exponential':
        gamma = cfg.get('gamma', 0.1)
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_name == 'cosine':
        T_max = cfg.get('T_max', 10)
        eta_min = cfg.get('eta_min', 0)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_name == 'plateau':
        mode = cfg.get('mode', 'max')
        factor = cfg.get('factor', 0.1)
        patience = cfg.get('patience', 10)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor,
                                                   patience=patience, verbose=True)
    
    else:
        return None
