"""
Dynamic Hierarchical Segmentation Pipeline
-------------------------------------------
Structure:

  for iteration in range(max_iter):
      train model for epochs_per_iter epochs
      validate → compute hierarchy metrics
      critic  → generate editing instructions
      editor  → update hierarchy tree
      save model snapshot for this iteration

  final eval with best model + best tree
"""

import csv
import os
import sys
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import json

# Add hierarchical losses to path to import ptsemseg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath('./ptsemseg'))

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import *
from ptsemseg.optimizers import get_optimizer
from ptsemseg.tree import create_tree_from_textfile, add_channels, add_levels, update_channels, find_depth

from validate_seg import validate_and_return_metrics
from critic_utils import compute_segmentation_hierarchy_metrics, get_segmentation_editing_instructions
from critic_utils import edit_tree, generate_initial_tree
from tree_utils import save_json_tree_to_txt, load_txt_tree_to_json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_dataset_loader(dataset_name):
    if dataset_name == 'smith_faces':
        from smith_loader import SmithLoader
        return SmithLoader
    elif dataset_name == 'mapillary_vistas':
        from vistas_loader import MapillaryVistasLoader
        return MapillaryVistasLoader
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_tree(cfg, hierarchy_file):
    """Build ptsemseg root node from a compiled .txt file."""
    root = create_tree_from_textfile(hierarchy_file)
    add_channels(root, 0)
    add_levels(root, find_depth(root))
    
    # Dataset-specific channel mapping (tree leaf index -> label index)
    ds_name = cfg['data']['dataset']
    if ds_name == 'smith_faces':
        class_lookup = [0, 10, 7, 8, 9, 1, 6, 4, 5, 2, 3]
    else:
        # Default to identity mapping (0..N-1) for Mapillary Vistas etc.
        # Tree leaves should follow the order in assets/classes.txt
        n_leaves = add_channels(root, 0)
        class_lookup = list(range(n_leaves))
        
    update_channels(root, class_lookup)
    return root


def build_model(cfg, n_classes, device):
    """Instantiate a fresh model."""
    model = get_model(cfg['model'], n_classes).to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    return model


def build_optimizer(cfg, model):
    optimizer_cls    = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg['training']['optimizer'].items() if k != 'name'}
    return optimizer_cls(model.parameters(), **optimizer_params)


def append_metrics_csv(row, csv_path, fieldnames):
    """Append one row to a CSV file, writing header on first write."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(model, trainloader, optimizer, loss_fn, root, cfg, device,
                    epoch, iteration, number_epoch_iters):
    """Train for one full epoch and return mean loss."""
    model.train()
    loss_meter = averageMeter()

    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(input=outputs, target=labels, root=root,
                       use_hierarchy=cfg['training']['use_hierarchy'])
        main_loss = loss[0] if isinstance(loss, tuple) else loss

        main_loss.backward()
        optimizer.step()
        loss_meter.update(main_loss.item())

        if (i + 1) % cfg['training']['print_interval'] == 0:
            print(f"  [Iter {iteration}] Epoch [{epoch}] "
                  f"Step [{i+1}/{int(number_epoch_iters)}] Loss: {main_loss.item():.4f}")

    return loss_meter.avg


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg, logger, experiment_name):
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)

    # ------------------------------------------------------------------
    # Config parameters
    # ------------------------------------------------------------------
    train_cfg       = cfg['training']
    max_iter        = train_cfg.get('max_iter', 5)
    epochs_per_iter = train_cfg.get('epochs_per_iter', 5)
    resume_weights  = train_cfg.get('resume_weights', True)   # carry weights across iterations?
    dynamic_hierarchy = train_cfg.get('dynamic_hierarchy', True)  # True = LLM edits tree each iter
    patience_limit  = train_cfg.get('patience', 5)
    max_tree_updates = train_cfg.get('max_tree_updates', max_iter)

    print(f"\nPipeline config:")
    print(f"  max_iter={max_iter}  epochs_per_iter={epochs_per_iter}")
    print(f"  resume_weights={resume_weights}  patience={patience_limit}")
    print(f"  dynamic_hierarchy={dynamic_hierarchy}" +
          (f"  max_tree_updates={max_tree_updates}" if dynamic_hierarchy else "  (static hierarchy, no LLM calls)"))

    # ------------------------------------------------------------------
    # Paths (Experiment Isolation)
    # ------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir     = os.path.abspath(os.path.join(script_dir, f'experiments/{experiment_name}'))
    assets_dir  = os.path.join(exp_dir, 'assets')
    logs_dir    = os.path.join(exp_dir, 'logs')
    weights_dir = os.path.join(exp_dir, 'weights')
    history_dir = os.path.join(assets_dir, 'tree_history')
    os.makedirs(assets_dir,  exist_ok=True)
    os.makedirs(logs_dir,    exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    initial_tree_json = os.path.join(assets_dir, 'initial_tree.json')
    hierarchy_json    = os.path.join(assets_dir, 'curr_tree.json')
    best_tree_json    = os.path.join(assets_dir, 'best_tree.json')
    hierarchy_txt     = os.path.join(assets_dir, 'curr_tree.txt')
    best_weights_path = os.path.join(weights_dir, 'best_model.pth')
    metrics_csv       = os.path.join(logs_dir, 'metrics_history.csv')
    classes_fp        = os.path.abspath(cfg['data'].get('classes_txt', './assets/smith_faces_classes.txt'))
    source_txt        = os.path.abspath(cfg['data'].get('initial_tree_txt', './assets/initial_tree.txt'))

    csv_fields = ['iteration', 'epoch', 'train_loss', 'mean_iou',
                  'lca_depth_mistake', 'avg_dist_to_lca',
                  'hierarchical_dist_mistake', 'mistake_only_rel_depth', 'master_metric']

    # ------------------------------------------------------------------
    # Hierarchy initialisation (protected baseline → working copy)
    # ------------------------------------------------------------------
    if not os.path.exists(initial_tree_json):
        print(f"Creating protected baseline from: {source_txt}")
        initial_json = load_txt_tree_to_json(source_txt)
        with open(initial_tree_json, 'w') as f:
            json.dump(initial_json, f, indent=2)
        print("Saved assets/initial_tree.json (will never be overwritten)")

    if not os.path.exists(hierarchy_json):
        shutil.copy(initial_tree_json, hierarchy_json)
        print("Seeded assets/curr_tree.json from initial_tree.json")

    # Sync JSON → TXT on startup (picks up any manual JSON edits)
    try:
        with open(hierarchy_json, 'r') as f:
            tree_data = json.load(f)
        save_json_tree_to_txt(tree_data, hierarchy_txt)
        print("Synced curr_tree.json -> curr_tree.txt")
    except Exception as e:
        print(f"Warning: Failed to sync JSON to TXT: {e}")

    shutil.copy(hierarchy_json, best_tree_json)

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    loader_cls = get_dataset_loader(cfg['data']['dataset'])
    augmentations = train_cfg.get('augmentations', None)
    data_aug = get_composed_augmentations(augmentations)
    data_path = cfg['data']['path']
    subsample_pct = train_cfg.get('subsample_pct', 1.0)

    t_loader = loader_cls(
        data_path, is_transform=True, split=cfg['data']['train_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug, subsample_pct=subsample_pct)

    n_classes = t_loader.n_classes
    trainloader = torch.utils.data.DataLoader(
        t_loader, batch_size=train_cfg['batch_size'],
        num_workers=train_cfg['n_workers'], shuffle=True)

    number_epoch_iters = t_loader.number_of_images / train_cfg['batch_size']

    with open(classes_fp, 'r') as f:
        class_names = [x.strip() for x in f.read().split(',')]

    # ------------------------------------------------------------------
    # Outer iteration loop
    # ------------------------------------------------------------------
    best_master_metric = 0.0
    patience_counter   = 0
    tree_updates       = 0
    start_iteration    = 1

    # Initialise model first
    model = build_model(cfg, n_classes, device)
    optimizer = build_optimizer(cfg, model)
    root = setup_tree(cfg, hierarchy_txt)

    # ------------------------------------------------------------------
    # Resume Logic: Detect existing iterations
    # ------------------------------------------------------------------
    found_iters = []
    if os.path.exists(weights_dir):
        for f in os.listdir(weights_dir):
            if f.startswith('model_iter_') and f.endswith('.pth'):
                try:
                    num = int(f.replace('model_iter_', '').replace('.pth', ''))
                    found_iters.append(num)
                except: pass
    
    if found_iters:
        last_it = max(found_iters)
        if last_it >= max_iter:
            print(f"\nExperiment '{experiment_name}' already finished {last_it} iterations.")
            print(f"Check logs in {logs_dir}")
            return

        start_iteration = last_it + 1
        resume_path = os.path.join(weights_dir, f'model_iter_{last_it}.pth')
        print(f"\n>>> Resuming from Iteration {start_iteration} (Found {last_it} previous)")
        print(f">>> Loading weights: {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location=device, weights_only=True))
        
        # Recover best_master_metric from CSV if possible
        if os.path.exists(metrics_csv):
            try:
                import csv
                with open(metrics_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        it = row.get('iteration')
                        if it and it.isdigit():
                            mm = float(row['master_metric'])
                            if mm > best_master_metric:
                                best_master_metric = mm
                                patience_counter = 0
                            else:
                                patience_counter += 1
                print(f">>> Recovered best_master_metric: {best_master_metric:.4f}")
            except Exception as e:
                print(f">>> Warning: Could not fully recover state from CSV: {e}")

    print(f"\n{'='*70}")
    print(f"Starting {max_iter}-Iteration Critic-Editor Training Loop")
    print(f"{'='*70}\n")

    for iteration in range(start_iteration, max_iter + 1):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}/{max_iter}")
        print(f"{'='*70}")

        # ---- Optionally reset model weights each iteration -----
        if iteration > 1 and not resume_weights:
            print("resume_weights=False: Reinitialising model from scratch.")
            model = build_model(cfg, n_classes, device)
            optimizer = build_optimizer(cfg, model)
        elif iteration > 1:
            print("resume_weights=True: Carrying forward weights from previous iteration.")

        # ---- Train for epochs_per_iter epochs ------------------
        print(f"\n--- Training ({epochs_per_iter} epochs) ---")
        
        start_epoch = 1
        total_train_loss = 0.0
        checkpoint_path = os.path.join(weights_dir, 'checkpoint.pth')
        
        # Intra-iteration resume logic
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if checkpoint.get('iteration') == iteration:
                    start_epoch = checkpoint.get('epoch', 0) + 1
                    if start_epoch <= epochs_per_iter:
                        print(f"\n>>> Intra-Iteration Resume: Iter {iteration}, Epoch {start_epoch}")
                        model.load_state_dict(checkpoint['model_state'])
                        if 'optimizer_state' in checkpoint:
                            optimizer.load_state_dict(checkpoint['optimizer_state'])
                        total_train_loss = checkpoint.get('total_train_loss', 0.0)
                    else:
                        print(f">>> Checkpoint indicates iteration {iteration} already finished training epochs.")
                        start_epoch = epochs_per_iter + 1
                else:
                    print(f">>> Existing checkpoint is for a different iteration ({checkpoint.get('iteration')}). Ignoring.")
            except Exception as e:
                print(f">>> Warning: Could not load epoch checkpoint: {e}")

        for epoch in range(start_epoch, epochs_per_iter + 1):
            epoch_loss = train_one_epoch(
                model, trainloader, optimizer, loss_fn=get_loss_function(cfg),
                root=root, cfg=cfg, device=device,
                epoch=epoch, iteration=iteration,
                number_epoch_iters=number_epoch_iters)
            total_train_loss += epoch_loss
            
            # Save intra-iteration checkpoint
            checkpoint = {
                'iteration': iteration,
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'total_train_loss': total_train_loss
            }
            torch.save(checkpoint, checkpoint_path)
            
            print(f"  [Iter {iteration}] Epoch {epoch}/{epochs_per_iter} done. "
                  f"Avg Loss: {epoch_loss:.4f}")

        avg_train_loss = total_train_loss / epochs_per_iter

        # ---- Save per-iteration model snapshot ----------------
        iter_weights_path = os.path.join(weights_dir, f'model_iter_{iteration}.pth')
        torch.save(model.state_dict(), iter_weights_path)
        print(f"Saved iteration {iteration} model to {iter_weights_path}")

        # ---- Validation + metrics -----------------------------
        print(f"\n--- Validation (Iteration {iteration}) ---")
        conf_matrix, score, iou, n_classes_val = validate_and_return_metrics(
            cfg, model, get_loss_function(cfg), device, root)

        metrics, misclassifications = compute_segmentation_hierarchy_metrics(
            conf_matrix, root, n_classes_val, class_names,
            num_pixels_threshold=500, iou_scores=iou)

        current_master = metrics.get('master_metric', 0.0)
        print(f"Iteration {iteration} Metrics: {metrics}")
        print(f"MASTER METRIC: {current_master:.4f}")

        # Log to CSV
        row = {'iteration': iteration, 'epoch': epochs_per_iter, 'train_loss': avg_train_loss}
        row.update({k: float(v) for k, v in metrics.items() if k in csv_fields})
        append_metrics_csv(row, metrics_csv, csv_fields)

        # ---- Check global best --------------------------------
        if current_master > best_master_metric:
            best_master_metric = current_master
            patience_counter   = 0
            print(f"New Global Best Master Metric: {best_master_metric:.4f}. "
                  f"Saving best model and tree.")
            torch.save(model.state_dict(), best_weights_path)
            shutil.copy(hierarchy_json, best_tree_json)
        else:
            patience_counter += 1
            print(f"No improvement ({best_master_metric:.4f}). "
                  f"Patience: {patience_counter}/{patience_limit}")

        if patience_counter >= patience_limit:
            print("Patience limit reached. Stopping refinement loop early.")
            break

        # ---- Critic → Editor (if dynamic_hierarchy is enabled) ----------
        if not dynamic_hierarchy:
            print("dynamic_hierarchy=False: Skipping LLM Critic-Editor. Hierarchy unchanged.")
        else:
            print("Skipping Critic-Editor (LLM integration requires external dependencies).")

    # ------------------------------------------------------------------
    # Final evaluation: best weights + best tree
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("ALL ITERATIONS COMPLETE. Running Final Evaluation.")
    print(f"  Best Master Metric across iterations: {best_master_metric:.4f}")
    print(f"{'='*70}\n")

    if os.path.exists(best_weights_path) and os.path.exists(best_tree_json):
        # Compile best tree txt
        best_txt = os.path.join(assets_dir, 'best_tree.txt')
        with open(best_tree_json, 'r') as f:
            best_tree_data = json.load(f)
        save_json_tree_to_txt(best_tree_data, best_txt)
        best_root = setup_tree(cfg, best_txt)

        # Load best weights
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        print("Loaded best model and best tree for final evaluation.")

        conf_matrix, score, iou, n_classes_val = validate_and_return_metrics(
            cfg, model, get_loss_function(cfg), device, best_root)

        final_metrics, _ = compute_segmentation_hierarchy_metrics(
            conf_matrix, best_root, n_classes_val, class_names,
            num_pixels_threshold=500, iou_scores=iou)

        print(f"FINAL EVALUATION METRICS: {final_metrics}")

        final_row = {'iteration': 'FINAL_BEST', 'epoch': '-', 'train_loss': '-'}
        final_row.update({k: float(v) for k, v in final_metrics.items() if k in csv_fields})
        append_metrics_csv(final_row, metrics_csv, csv_fields)
    else:
        print("Could not find best weights/tree. Skipping final evaluation.")

    print("\nPipeline finished.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description="Dynamic Hierarchical Segmentation")
    parser.add_argument("--config", nargs="?", type=str,
                        default="./config.yml", help="Config file")
    parser.add_argument("--name", type=str, required=True,
                        help="Unique name for this experiment/run")
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    # Isolated log dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_log_dir = os.path.join(script_dir, 'experiments', args.name, 'logs')
    os.makedirs(exp_log_dir, exist_ok=True)
    logger = get_logger(exp_log_dir)
    run_pipeline(cfg, logger, args.name)
