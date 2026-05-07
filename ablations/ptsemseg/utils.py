"""
Utility functions for semantic segmentation.
Includes logging, file utilities, and common helpers.
"""

import os
import glob
import logging
from typing import List


def get_logger(logfile: str = None, loglevel: int = logging.DEBUG) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        logfile: Optional file path to write logs to
        loglevel: Logging level (default: DEBUG)
    
    Returns:
        logger: Configured logger instance
    """
    logger = logging.getLogger('ptsemseg')
    logger.setLevel(loglevel)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler (if specified)
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setLevel(loglevel)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def recursive_glob(rootdir: str, suffix: str = '*') -> List[str]:
    """
    Recursively find all files matching a suffix in a directory.
    
    Args:
        rootdir: Root directory to search
        suffix: File suffix/extension to match (e.g., '.jpg', '.png')
    
    Returns:
        List of file paths matching the suffix
    """
    if not suffix.startswith('*'):
        suffix = '*' + suffix
    
    pattern = os.path.join(rootdir, '**', suffix)
    return sorted(glob.glob(pattern, recursive=True))


def ensure_dir(path: str):
    """
    Ensure a directory exists, create if necessary.
    
    Args:
        path: Directory path
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_checkpoint(state: dict, filepath: str):
    """
    Save a checkpoint dictionary to file.
    
    Args:
        state: Dictionary with checkpoint data
        filepath: Path to save checkpoint to
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    import torch
    torch.save(state, filepath)


def load_checkpoint(filepath: str, device='cpu'):
    """
    Load a checkpoint from file.
    
    Args:
        filepath: Path to checkpoint file
        device: Device to load to ('cpu' or 'cuda')
    
    Returns:
        Loaded checkpoint dictionary
    """
    import torch
    return torch.load(filepath, map_location=device)


def count_parameters(model) -> int:
    """
    Count total number of trainable parameters in model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DummyLogger:
    """Dummy logger that does nothing (fallback if logging not configured)."""
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass
