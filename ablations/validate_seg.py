import os
import torch
from tqdm import tqdm
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations

def get_dataset_loader(dataset_name):
    if dataset_name == 'smith_faces':
        from smith_loader import SmithLoader
        return SmithLoader
    elif dataset_name == 'mapillary_vistas':
        from vistas_loader import MapillaryVistasLoader
        return MapillaryVistasLoader
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def validate_and_return_metrics(cfg, model_tree, loss_fn, device, root):
    """
    Modified validation function that returns metrics and confusion matrix 
    for the hierarchy (tree) model.
    """
    data_path = cfg['data']['path']
    loader_cls = get_dataset_loader(cfg['data']['dataset'])
    augmentations = cfg['training'].get('augmentations', None)
    data_aug = get_composed_augmentations(augmentations)
    
    v_loader = loader_cls(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug)
    
    n_classes = v_loader.n_classes
    valloader = torch.utils.data.DataLoader(v_loader, 
                                            batch_size=cfg['training']['batch_size'], 
                                            num_workers=cfg['training']['n_workers'])
    
    running_metrics_val_tree = runningScore(n_classes)
    
    model_tree.eval()
    with torch.no_grad():
        print("Validation loop...")
        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader), total=min(len(valloader), 200)):
            # NOTE: limiting validation to 200 batches to speed up critic loop
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)

            outputs_tree = model_tree(images_val)
            pred_tree = outputs_tree.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            
            running_metrics_val_tree.update(gt, pred_tree)  # updates confusion matrix
            
            if i_val > 100: # Fast dev val
                break

        score_tree, class_iou_tree = running_metrics_val_tree.get_scores()
        confusion_matrix = running_metrics_val_tree.confusion_matrix.copy()

    return confusion_matrix, score_tree, class_iou_tree, n_classes
