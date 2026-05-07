"""
PyTorch Semantic Segmentation Library (ptsemseg)
Custom semantic segmentation utilities for hierarchical dense prediction.
"""

from .tree import (
    TreeNode,
    find_depth,
    create_tree_from_textfile,
    add_channels,
    add_levels,
    update_channels,
    get_leaf_nodes,
    get_all_nodes,
    build_tree_from_json,
    get_node_by_name,
    get_lca,
    get_distance,
)

from .metrics import (
    averageMeter,
    runningScore,
    compute_iou,
)

from .augmentations import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    RandomGaussianBlur,
    RandomCrop,
    RandomBrightness,
    Identity,
    get_composed_augmentations,
)

from .utils import (
    get_logger,
    recursive_glob,
    ensure_dir,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
)

from .optimizers import (
    get_optimizer,
    get_scheduler,
)

from .models import (
    get_model,
    UNet,
    DeepLabV3,
    FCN,
)

from .loss import (
    get_loss_function,
    CrossEntropyLoss,
    TreeAwareLoss,
    FocalLoss,
    DiceLoss,
    CombinedLoss,
)

__version__ = '0.1.0'

__all__ = [
    # Tree
    'TreeNode',
    'find_depth',
    'create_tree_from_textfile',
    'add_channels',
    'add_levels',
    'update_channels',
    'get_leaf_nodes',
    'get_all_nodes',
    'build_tree_from_json',
    'get_node_by_name',
    'get_lca',
    'get_distance',
    
    # Metrics
    'averageMeter',
    'runningScore',
    'compute_iou',
    
    # Augmentations
    'Compose',
    'RandomHorizontalFlip',
    'RandomVerticalFlip',
    'RandomRotation',
    'RandomGaussianBlur',
    'RandomCrop',
    'RandomBrightness',
    'Identity',
    'get_composed_augmentations',
    
    # Utils
    'get_logger',
    'recursive_glob',
    'ensure_dir',
    'save_checkpoint',
    'load_checkpoint',
    'count_parameters',
    
    # Optimizers
    'get_optimizer',
    'get_scheduler',
    
    # Models
    'get_model',
    'UNet',
    'DeepLabV3',
    'FCN',
    
    # Loss
    'get_loss_function',
    'CrossEntropyLoss',
    'TreeAwareLoss',
    'FocalLoss',
    'DiceLoss',
    'CombinedLoss',
]
