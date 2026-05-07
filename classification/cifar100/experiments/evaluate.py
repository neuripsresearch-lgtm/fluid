import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import timm
import argparse
import json
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform, pdist
from tqdm import tqdm

# ==========================================
# 1. Tree & Hierarchy Helper Functions
# ==========================================

class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []

def build_tree(data, parent=None):
    """Recursively builds the tree from JSON data."""
    if isinstance(data, str):
        node = Node(data.replace(' ', '_'), parent)
        return node
    
    node_name = data.get('name', 'Unnamed').replace(' ', '_')
    node = Node(node_name, parent)
    
    if 'children' in data and data['children']:
        for child_data in data['children']:
            child_node = build_tree(child_data, node)
            node.add_child(child_node)
    return node

def find_node(root, name):
    """Finds a node by name."""
    if root.name == name: return root
    for child in root.children:
        res = find_node(child, name)
        if res: return res
    return None

def get_depth(node):
    """Returns depth of node (root=0)."""
    d = 0
    curr = node
    while curr.parent:
        d += 1
        curr = curr.parent
    return d

def find_lca(node1, node2):
    """Finds Lowest Common Ancestor."""
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

def get_tree_distance(node1, node2):
    """Calculates hops between two nodes."""
    lca = find_lca(node1, node2)
    if not lca: return 100 # Should not happen in connected tree
    d1 = get_depth(node1)
    d2 = get_depth(node2)
    lca_d = get_depth(lca)
    return d1 + d2 - 2 * lca_d

def get_average_leaf_depth(root):
    leaves = []
    def _collect(n):
        if not n.children: leaves.append(n)
        else:
            for c in n.children: _collect(c)
    _collect(root)
    return sum(get_depth(l) for l in leaves) / len(leaves)

# ==========================================
# 2. Metric Calculation Logic
# ==========================================

def calculate_alignment(model, loader, device, idx_to_class, tree_root, classes):
    """
    Calculates Tree-Visual Alignment (Spearman Rho).
    1. Extract features for all test images.
    2. Compute Class Centroids (average feature per class).
    3. Compute Visual Distance Matrix (Cosine).
    4. Compute Tree Distance Matrix (Hops).
    5. Correlate.
    """
    print("Extracting features for Tree-Visual Alignment...")
    model.eval()
    
    # Store features: {class_name: [tensor, tensor, ...]}
    class_features = {c: [] for c in classes}
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Feature Extraction"):
            images = images.to(device)
            # Forward pass through backbone (features)
            # Swin-T output before head is (B, 768)
            features = model.forward_features(images) 
            if len(features.shape) > 2: # Global pool if needed (Swin usually outputs (B, 768) after pooling in forward_features for timm)
                features = features.mean(dim=1) 
            
            for i in range(len(labels)):
                lbl = labels[i].item()
                c_name = idx_to_class[lbl]
                class_features[c_name].append(features[i].cpu().numpy())

    # Compute Centroids
    centroids = []
    valid_classes = [] # Ensure we follow the order of 'classes' list
    for c in classes:
        if class_features[c]:
            # Average all feature vectors for this class
            mean_vec = np.mean(np.vstack(class_features[c]), axis=0)
            centroids.append(mean_vec)
            valid_classes.append(c)
    
    centroids = np.array(centroids) # (100, 768)
    
    # 1. Visual Distance Matrix (Cosine Distance)
    # pdist computes pairwise distances
    print("Computing Visual Distance Matrix...")
    vis_dists_condensed = pdist(centroids, metric='cosine')
    
    # 2. Tree Distance Matrix
    print("Computing Tree Distance Matrix...")
    n = len(valid_classes)
    tree_dists = np.zeros((n, n))
    
    # Cache nodes
    nodes = [find_node(tree_root, c) for c in valid_classes]
    
    for i in range(n):
        for j in range(i + 1, n):
            d = get_tree_distance(nodes[i], nodes[j])
            tree_dists[i, j] = d
            tree_dists[j, i] = d
            
    tree_dists_condensed = squareform(tree_dists)

    # 3. Spearman Correlation
    print("Calculating Correlation...")
    rho, p_val = spearmanr(vis_dists_condensed, tree_dists_condensed)
    
    return rho

# ==========================================
# 3. Main Evaluation Loop
# ==========================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    testset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Standardize class names (replace spaces with underscores)
    classes = [c.replace(' ', '_') for c in testset.classes]
    idx_to_class = {i: c for i, c in enumerate(classes)}

    # --- Load Tree ---
    with open(args.hierarchy_path, 'r') as f:
        tree_data = json.load(f)
    root = build_tree(tree_data)
    avg_leaf_depth = get_average_leaf_depth(root)
    print(f"Hierarchy Loaded. Avg Leaf Depth: {avg_leaf_depth:.4f}")

    # --- Load Model ---
    print(f"Loading Model from {args.model_path}...")
    # NOTE: Assuming Swin-Tiny architecture based on context. Change if using ResNet.
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=100)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Vars for Metrics ---
    total = 0
    correct = 0
    
    # Severity Accumulators
    mistake_count = 0
    sum_lca_depth_mistakes = 0.0
    sum_dist_to_lca_mistakes = 0.0
    sum_rel_depth_mistakes = 0.0
    
    # Overall Accumulator
    sum_rel_depth_all = 0.0

    # --- Evaluation Loop ---
    print("Starting Inference...")
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="Eval"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                total += 1
                true_idx = labels[i].item()
                pred_idx = preds[i].item()
                
                true_name = idx_to_class[true_idx]
                pred_name = idx_to_class[pred_idx]
                
                # Nodes
                node_true = find_node(root, true_name)
                node_pred = find_node(root, pred_name)
                
                if not node_true or not node_pred:
                    print(f"Warning: Node not found in tree: {true_name} or {pred_name}")
                    continue

                # Common Calcs
                lca = find_lca(node_true, node_pred)
                lca_d = get_depth(lca)
                
                # --- Metric 3: Relative LCA Depth (Overall) ---
                # For correct prediction, lca_d == leaf_depth, so ratio is 1.0 (Best)
                rel_depth = lca_d / avg_leaf_depth
                sum_rel_depth_all += rel_depth

                if true_idx == pred_idx:
                    correct += 1
                else:
                    # --- Mistake Logic ---
                    mistake_count += 1
                    
                    # Metric 1: LCA Depth
                    sum_lca_depth_mistakes += lca_d
                    
                    # Metric 2: Avg Dist to LCA
                    # dist(node, lca) = depth(node) - depth(lca)
                    d_true = get_depth(node_true) - lca_d
                    d_pred = get_depth(node_pred) - lca_d
                    avg_dist = (d_true + d_pred) / 2.0
                    sum_dist_to_lca_mistakes += avg_dist
                    
                    # Mistake-Only Rel Depth
                    sum_rel_depth_mistakes += rel_depth

    # --- Calculate Alignment ---
    alignment_score = calculate_alignment(model, testloader, device, idx_to_class, root, classes)

    # --- Finalize Metrics ---
    acc = 100 * correct / total
    
    # Avoid div by zero
    if mistake_count > 0:
        avg_lca_depth_m = sum_lca_depth_mistakes / mistake_count
        avg_dist_lca_m = sum_dist_to_lca_mistakes / mistake_count
        avg_rel_depth_m = sum_rel_depth_mistakes / mistake_count
    else:
        avg_lca_depth_m, avg_dist_lca_m, avg_rel_depth_m = 0, 0, 0
        
    avg_rel_depth_all = sum_rel_depth_all / total

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Top-1 Accuracy (%):            {acc:.2f}")
    print(f"LCA Depth (Metric 1):          {avg_lca_depth_m:.4f}  (Higher is better for severity)")
    print(f"Avg Dist to LCA (Metric 2):    {avg_dist_lca_m:.4f}  (Lower is better)")
    print(f"Relative LCA Depth (Metric 3): {avg_rel_depth_all:.4f}  (Overall score, 1.0 is perfect)")
    print(f"Mistake-Only Rel Depth:        {avg_rel_depth_m:.4f}  (Higher is better)")
    print(f"Tree-Visual Alignment:         {alignment_score:.4f}  (Spearman Rho)")
    print("="*50 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to .pth file')
    parser.add_argument('--hierarchy-path', type=str, required=True, help='Path to .json hierarchy')
    parser.add_argument('--data-path', type=str, default='./data', help='Path to CIFAR100 data')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    
    main(args)