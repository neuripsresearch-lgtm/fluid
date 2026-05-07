import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import timm
import argparse
import os
import json
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Subset

# --- Reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# --- Tree Helper Classes ---
class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
    def add_child(self, child_node):
        self.children.append(child_node)

def build_tree(data, parent=None):
    if isinstance(data, str):
        node = Node(data.replace(' ', '_'), parent)
        return node
    node = Node(data.get('name', 'Unnamed').replace(' ', '_'), parent)
    if 'children' in data and data['children']:
        for child_data in data['children']:
            child_node = build_tree(child_data, node)
            node.add_child(child_node)
    return node

def find_node(root, name):
    if root.name == name: return root
    for child in root.children:
        found = find_node(child, name)
        if found: return found
    return None

def get_depth(node):
    d = 0
    curr = node
    while curr.parent:
        d += 1
        curr = curr.parent
    return d

def get_lca(node1, node2):
    path1 = set()
    curr = node1
    while curr:
        path1.add(curr)
        curr = curr.parent
    curr = node2
    while curr:
        if curr in path1:
            return curr
        curr = curr.parent
    return None

def get_lca_depth(node1, node2):
    lca = get_lca(node1, node2)
    return get_depth(lca) if lca else 0

def get_tree_distance(node1, node2):
    """Calculates distance between two nodes: dist(a, b) = depth(a) + depth(b) - 2*depth(lca)"""
    d1 = get_depth(node1)
    d2 = get_depth(node2)
    lca_d = get_lca_depth(node1, node2)
    return d1 + d2 - 2 * lca_d

def get_average_leaf_depth(root):
    leaves = []
    def _collect(n):
        if not n.children: leaves.append(n)
        else:
            for c in n.children: _collect(c)
    _collect(root)
    if not leaves: return 1.0
    return sum(get_depth(l) for l in leaves) / len(leaves)

# --- Hierarchical Loss Logic (Soft Labels) ---
class HierarchicalSoftLoss(nn.Module):
    def __init__(self, hierarchy_file, classes, beta=1.0, device='cuda'):
        super().__init__()
        self.beta = beta
        self.device = device
        self.num_classes = len(classes)
        
        print(f"Initializing Hierarchical Soft Labels (beta={beta})...")
        
        # 1. Load Tree
        with open(hierarchy_file, 'r') as f:
            tree_data = json.load(f)
        self.root = build_tree(tree_data)
        self.avg_leaf_depth = get_average_leaf_depth(self.root)
        
        # 2. Map class indices to Tree Nodes
        self.nodes = []
        self.idx_to_node = {} # Helper for validation
        for idx, class_name in enumerate(classes):
            clean_name = class_name.replace(' ', '_')
            node = find_node(self.root, clean_name)
            if not node:
                print(f"Warning: Class '{clean_name}' not found in hierarchy! using fallback.")
            self.nodes.append(node)
            self.idx_to_node[idx] = node
            
        # 3. Pre-compute Distance Matrix (N x N)
        dist_matrix = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i == j:
                    dist_matrix[i, j] = 0
                elif self.nodes[i] and self.nodes[j]:
                    dist_matrix[i, j] = get_tree_distance(self.nodes[i], self.nodes[j])
                else:
                    dist_matrix[i, j] = 100 # High penalty for missing nodes
        
        # 4. Convert Distances to Soft Probabilities
        # Formula: P(y=k | target=i) = exp(-beta * dist(i, k)) / Z
        neg_dist = -self.beta * dist_matrix
        # Numerical stability: subtract max per row
        neg_dist = neg_dist - np.max(neg_dist, axis=1, keepdims=True)
        exp_dist = np.exp(neg_dist)
        self.soft_labels = exp_dist / np.sum(exp_dist, axis=1, keepdims=True)
        
        self.soft_labels = torch.tensor(self.soft_labels, dtype=torch.float32).to(device)

    def forward(self, logits, targets):
        target_dist = self.soft_labels[targets]
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.kl_div(log_probs, target_dist, reduction='batchmean')
        return loss

def freeze_all_except_last_two_layers(model):
    """
    Freezes all layers except:
      1. The Classifier Head
      2. The Final Normalization
      3. The Last Two Backbone Stages (Blocks)
    
    This ensures a significant portion of the model (likely 30-50% of params) 
    is trainable for refinement.
    """
    print("Freezing all layers except head, norm, and last two backbone stages...")
    
    # 1. Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    trainable_params = []

    # 2. Unfreeze Head
    if hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        print(" -> Unfroze 'head' layer.")
    elif hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        print(" -> Unfroze 'fc' layer.")
    
    # 3. Unfreeze Norm
    if hasattr(model, 'norm'):
        for param in model.norm.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        print(" -> Unfroze 'norm' layer.")

    # 4. Unfreeze Last Two Backbone Stages
    # Swin Transformer (timm) stores stages in 'layers'
    if hasattr(model, 'layers'):
        num_layers = len(model.layers)
        
        # Unfreeze last stage (e.g., stage 4)
        if num_layers > 0:
            for param in model.layers[-1].parameters():
                param.requires_grad = True
                trainable_params.append(param)
            print(f" -> Unfroze backbone stage {num_layers} (layers[-1]).")

        # Unfreeze second-to-last stage (e.g., stage 3)
        # COMMENTED FOR NEW EXPERIMENT
        # if num_layers > 1:
        #     for param in model.layers[-2].parameters():
        #         param.requires_grad = True
        #         trainable_params.append(param)
        #     print(f" -> Unfroze backbone stage {num_layers-1} (layers[-2]).")
            
    # Fallback for ViT (which uses 'blocks') or ResNet (which uses 'layer4', 'layer3')
    elif hasattr(model, 'blocks'): # ViT
        num_blocks = len(model.blocks)
        for i in range(max(0, num_blocks - 2), num_blocks):
            for param in model.blocks[i].parameters():
                param.requires_grad = True
                trainable_params.append(param)
        print(f" -> Unfroze last {min(2, num_blocks)} ViT blocks.")

    # Print trainable parameters summary
    num_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {num_trainable:,} / {total_params:,} ({100*num_trainable/total_params:.2f}%)")

def validate_master_metric(model, loader, device, criterion):
    """
    Calculates the Master Metric: Accuracy * Mistake-Only Relative LCA Depth.
    Returns: master_metric, accuracy, mistake_only_rel_depth
    """
    model.eval()
    
    correct_count = 0
    mistake_rel_depth_sum = 0.0
    mistake_count = 0
    total_samples = 0
    
    idx_to_node = criterion.idx_to_node
    avg_leaf_depth = criterion.avg_leaf_depth

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            for i in range(len(labels)):
                true_idx = labels[i].item()
                pred_idx = predicted[i].item()
                total_samples += 1

                if true_idx == pred_idx:
                    correct_count += 1
                else:
                    # Mistake processing
                    mistake_count += 1
                    node_t = idx_to_node.get(true_idx)
                    node_p = idx_to_node.get(pred_idx)
                    
                    if node_t and node_p:
                        lca_d = get_lca_depth(node_t, node_p)
                        if avg_leaf_depth > 0:
                            mistake_rel_depth_sum += (lca_d / avg_leaf_depth)
                    else:
                        # Fallback for missing nodes (should not happen if classes match)
                        pass

    if total_samples == 0: return 0.0, 0.0, 0.0
    
    # 1. Calculate Accuracy
    accuracy = correct_count / total_samples
    
    # 2. Calculate Mistake-Only Relative Depth
    avg_mistake_rel_depth = 0.0
    if mistake_count > 0:
        avg_mistake_rel_depth = mistake_rel_depth_sum / mistake_count
    else:
        # Edge case: No mistakes (100% accuracy). 
        # In this theoretical case, the metric should be high.
        avg_mistake_rel_depth = 1.0 
        
    # 3. Master Metric Calculation
    master_metric = accuracy * avg_mistake_rel_depth
    
    return master_metric, accuracy, avg_mistake_rel_depth

def main(args):
    set_seed(42) # Set seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Setup
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Transform for Validation (Standard deterministic crop)
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    print("Loading Dataset (Train Split Only)...")
    # Load FULL Training set
    full_trainset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
    
    # --- SPLIT INTO TRAIN AND VAL (Strict Isolation) ---
    num_train = len(full_trainset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train)) # 10% for validation
    
    # Shuffle indices deterministically based on seed 42 set at start
    np.random.shuffle(indices)
    
    train_idx, val_idx = indices[split:], indices[:split]
    
    print(f"Splitting Training Data: {len(train_idx)} Train / {len(val_idx)} Validation")

    # Create Subsets
    train_subset = Subset(full_trainset, train_idx)
    # IMPORTANT: Validation subset needs the VALIDATION transform (deterministic)
    # We cheat slightly by reloading the dataset with val transforms for the val subset
    full_valset_raw = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_val)
    val_subset = Subset(full_valset_raw, val_idx)

    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Note: Test loader is REMOVED from training script completely to prevent leakage.

    # 2. Model Setup
    print("Creating Swin Transformer (MBM)...")
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=100)
    model = model.to(device)

    # --- Transfer Learning / Resume Logic ---
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        try:
            model.load_state_dict(torch.load(args.resume, map_location=device))
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}. Starting from fresh pretrained.")
    else:
        print("Starting training from ImageNet pretrained weights.")

    # --- ITERATION-AWARE FREEZING LOGIC ---
    if args.iteration_num > 1:
        print(f"\n{'='*60}")
        print(f"Iteration {args.iteration_num} > 1")
        print(f"Refinement Mode: Freezing backbone, training last 2 layers only.")
        print(f"{'='*60}\n")
        freeze_all_except_last_two_layers(model)
    else:
        print(f"\n{'='*60}")
        print(f"Iteration {args.iteration_num} == 1")
        print(f"Initial Mode: Training ENTIRE backbone.")
        print(f"{'='*60}\n")

    # 3. Loss Function Setup
    criterion = HierarchicalSoftLoss(
        hierarchy_file=args.hierarchy_path,
        classes=full_trainset.classes,
        beta=args.beta,
        device=device
    )

    # Only optimize parameters that require gradients
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Training Loop with Patience (Using MASTER METRIC)
    print(f"Starting training for {args.epochs} epochs (Patience: {args.patience})...")
    
    best_metric = 0.0 # Tracking Master Metric (Acc * MistakeRelDepth)
    patience_counter = 0

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
        
        scheduler.step()

        # Validation Step (New Master Metric)
        val_metric, val_acc, val_mistake_depth = validate_master_metric(model, valloader, device, criterion)
        
        print(f"Epoch {epoch+1} Stats:")
        print(f"  -> Accuracy: {val_acc:.4f}")
        print(f"  -> Mistake-Only Rel Depth: {val_mistake_depth:.4f}")
        print(f"  -> MASTER METRIC (Acc*Mistake): {val_metric:.4f} (Best: {best_metric:.4f})")

        if val_metric > best_metric:
            best_metric = val_metric
            print(f"Metric improved. Saving model to {args.save_path}")
            torch.save(model.state_dict(), args.save_path)
            patience_counter = 0 
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"Patience limit ({args.patience}) reached. Stopping training early.")
            break

    print(f"Training finished. Best Val Master Metric: {best_metric:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='./data', type=str)
    parser.add_argument('--hierarchy-path', default='./assets/curr_tree.json', type=str)
    parser.add_argument('--save-path', default='./weights/mbm_model.pth', type=str)
    parser.add_argument('--resume', default=None, type=str, help='Path to load weights from')
    parser.add_argument('--epochs', default=20, type=int) 
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--weight-decay', default=0.05, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--patience', default=5, type=int, help='Epochs to wait for improvement before stopping')
    parser.add_argument('--iteration-num', default=1, type=int, help='Current iteration number in the pipeline')
    # freeze-after-iteration is redundant now due to hardcoded logic, but kept for arg compatibility
    parser.add_argument('--freeze-after-iteration', default=5, type=int)
    
    args = parser.parse_args()
    main(args)