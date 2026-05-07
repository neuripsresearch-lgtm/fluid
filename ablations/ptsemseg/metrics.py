"""
Metrics for semantic segmentation evaluation.
Includes running score tracking, IoU computation, and metric aggregation.
"""

import numpy as np
from typing import Tuple


class averageMeter:
    """
    Tracks average of metric across batches/iterations.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """Update with a new value and count."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class runningScore:
    """
    Computes IoU and other metrics from confusion matrix.
    Used for segmentation evaluation.
    """
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
    
    def _fast_hist(self, label_true: np.ndarray, label_pred: np.ndarray) -> np.ndarray:
        """
        Build confusion matrix from ground truth and predictions.
        """
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2
        ).reshape(self.n_classes, self.n_classes)
        return hist
    
    def update(self, label_true: np.ndarray, label_pred: np.ndarray):
        """
        Update confusion matrix with batch predictions.
        
        Args:
            label_true: Ground truth labels (flattened or batched)
            label_pred: Predicted labels (flattened or batched)
        """
        # Flatten if needed
        if label_true.ndim > 1:
            label_true = label_true.flatten()
        if label_pred.ndim > 1:
            label_pred = label_pred.flatten()
        
        self.confusion_matrix += self._fast_hist(label_true, label_pred)
    
    def get_scores(self) -> Tuple[dict, dict]:
        """
        Compute overall metrics and per-class IoU.
        
        Returns:
            scores: dict with 'Mean IoU', 'mIoU', 'pixAcc', 'Class Wise IOU'
            class_iou: dict with per-class IoU values
        """
        hist = self.confusion_matrix
        
        # True positives, false positives, false negatives
        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp
        
        # IoU per class
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = tp / (tp + fp + fn)
            iou[~np.isfinite(iou)] = 0  # Handle division by zero
        
        # Mean IoU
        mean_iou = np.nanmean(iou)
        
        # Pixel accuracy
        pixel_acc = tp.sum() / hist.sum() if hist.sum() > 0 else 0
        
        # Construct class-wise IOU dict
        class_iou = {}
        for i in range(self.n_classes):
            class_iou[i] = iou[i]
        
        scores = {
            'Mean IoU': mean_iou,
            'mIoU': mean_iou,
            'pixAcc': pixel_acc,
            'Class Wise IOU': class_iou
        }
        
        return scores, class_iou
    
    def reset(self):
        """Reset confusion matrix."""
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def compute_iou(confusion_matrix: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute mean IoU and per-class IoU from confusion matrix.
    
    Args:
        confusion_matrix: (n_classes, n_classes) confusion matrix
    
    Returns:
        mean_iou: float, averaged IoU across classes
        class_iou: (n_classes,) per-class IoU values
    """
    hist = confusion_matrix
    tp = np.diag(hist)
    fp = hist.sum(axis=0) - tp
    fn = hist.sum(axis=1) - tp
    
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = tp / (tp + fp + fn)
        iou[~np.isfinite(iou)] = 0
    
    mean_iou = np.nanmean(iou)
    return mean_iou, iou
