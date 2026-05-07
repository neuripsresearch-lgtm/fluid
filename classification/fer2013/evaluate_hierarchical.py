import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import timm
import argparse
import json
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform, pdist
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

# --- Tree Utils (Identical to previous) ---
class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.height = 0 
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
        res = find_node(child, name)
        if res: return res
    return None

def compute_node_heights(node):
    if not node.children:
        node.height = 0
        return 0
    max_h = 0
    for c in node.children:
        max_h = max(max_h, compute_node_heights(c))
    node.height = max_h + 1
    return node.height

def get_depth(node):
    d = 0
    curr = node
    while curr.parent:
        d += 1
        curr = curr.parent
    return d

def find_lca(node1, node2):
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
    lca = find_lca(node1, node2)
    if not lca: return 100
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
    if not leaves: return 0
    return sum(get_depth(l) for l in leaves) / len(leaves)

# --- Alignment Metric ---
def calculate_alignment(model, loader, device, idx_to_class, tree_root, classes):
    print("Extracting features for Tree-Visual Alignment...")
    model.eval()
    class_features = {c: [] for c in classes}
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Feature Extraction"):
            images = images.to(device)
            features = model.forward_features(images) 
            if len(features.shape) > 2: 
                features = features.mean(dim=1) 
            
            for i in range(len(labels)):
                lbl = labels[i].item()
                c_name = idx_to_class[lbl]
                class_features[c_name].append(features[i].cpu().numpy())

    centroids = []
    valid_classes = [] 
    for c in classes:
        if class_features[c]:
            mean_vec = np.mean(np.vstack(class_features[c]), axis=0)
            centroids.append(mean_vec)
            valid_classes.append(c)
    
    centroids = np.array(centroids)
    if len(centroids) < 2: return 0.0

    print("Computing Visual Distance Matrix...")
    vis_dists_condensed = pdist(centroids, metric='cosine')
    
    print("Computing Tree Distance Matrix...")
    n = len(valid_classes)
    tree_dists = np.zeros((n, n))
    nodes = [find_node(tree_root, c) for c in valid_classes]
    
    for i in range(n):
        for j in range(i + 1, n):
            d = get_tree_distance(nodes[i], nodes[j])
            tree_dists[i, j] = d
            tree_dists[j, i] = d
            
    tree_dists_condensed = squareform(tree_dists)

    print("Calculating Correlation...")
    rho, _ = spearmanr(vis_dists_condensed, tree_dists_condensed)
    return rho

def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # --- SPLIT LOGIC ---
    if args.split == 'test':
        print("LOADING: OFFICIAL TEST SET")
        # EDITED: Point to the 'test' folder
        test_dir = os.path.join(args.data_path, 'test')
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test data not found at {test_dir}")
        dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
    else:
        print("LOADING: VALIDATION SPLIT")
        # EDITED: Point to 'train' folder (to extract validation subset)
        train_dir = os.path.join(args.data_path, 'train')
        full_trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
        
        num_train = len(full_trainset)
        indices = list(range(num_train))
        split = int(np.floor(0.1 * num_train))
        
        np.random.shuffle(indices)
        val_idx = indices[:split]
        dataset = Subset(full_trainset, val_idx)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # EDITED: Get classes from the ImageFolder structure
    # We need a dummy instance to read class names if we are using Subset
    if args.split == 'test':
        classes = dataset.classes
    else:
        classes = full_trainset.classes
        
    classes = [c.replace(' ', '_') for c in classes] # Safety
    idx_to_class = {i: c for i, c in enumerate(classes)}

    # 2. Load Tree
    with open(args.hierarchy_path, 'r') as f:
        tree_data = json.load(f)
    root = build_tree(tree_data)
    compute_node_heights(root) 
    avg_leaf_depth = get_average_leaf_depth(root)
    print(f"Hierarchy Loaded. Avg Leaf Depth: {avg_leaf_depth:.4f}")

    # 3. Load Model
    print(f"Loading MBM Model from {args.model_path}...")
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=len(classes))
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # 4. Evaluation Loop
    total = 0
    correct = 0
    mistake_stats = {'lca_depth': 0, 'rel_depth': 0, 'dist': 0, 'count': 0}
    rel_depth_all = 0.0 
    mistake_height_sum = 0.0
    topk_height_sums = {1: 0.0, 5: 0.0}
    
    all_true_labels = []
    all_pred_labels = []
    class_pair_mistakes = {}

    print("Starting Inference...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Eval"):
            images = images.to(device)
            logits = model(images)
            _, topk_preds = logits.topk(5, dim=1) 
            
            for i in range(len(labels)):
                total += 1
                t_idx = labels[i].item()
                p_idx = topk_preds[i, 0].item()
                
                all_true_labels.append(t_idx)
                all_pred_labels.append(p_idx)
                
                node_t = find_node(root, idx_to_class[t_idx])
                node_p = find_node(root, idx_to_class[p_idx])
                
                if node_t and node_p:
                    lca = find_lca(node_t, node_p)
                    lca_d = get_depth(lca) if lca else 0
                    rel_d = lca_d / avg_leaf_depth if avg_leaf_depth > 0 else 0
                    rel_depth_all += rel_d
                    
                    if t_idx == p_idx:
                        correct += 1
                    else:
                        mistake_stats['count'] += 1
                        mistake_stats['lca_depth'] += lca_d
                        mistake_stats['rel_depth'] += rel_d
                        
                        d_t = get_depth(node_t)
                        d_p = get_depth(node_p)
                        mistake_stats['dist'] += (d_t + d_p - 2*lca_d) 

                        mistake_height_sum += lca.height if lca else 0
                        
                        true_name = idx_to_class[t_idx]
                        pred_name = idx_to_class[p_idx]
                        pair_key = (true_name, pred_name)
                        if pair_key not in class_pair_mistakes:
                            class_pair_mistakes[pair_key] = []
                        avg_dist_legacy = (d_t + d_p - 2*lca_d) / 2.0
                        class_pair_mistakes[pair_key].append((lca_d, avg_dist_legacy))

                if node_t:
                    curr_sample_k_sum = 0.0
                    for k in range(5):
                        curr_p_idx = topk_preds[i, k].item()
                        node_curr_p = find_node(root, idx_to_class[curr_p_idx])
                        if node_curr_p:
                            curr_lca = find_lca(node_t, node_curr_p)
                            if curr_lca:
                                curr_sample_k_sum += curr_lca.height
                        
                        rank = k + 1
                        if rank in topk_height_sums:
                            topk_height_sums[rank] += (curr_sample_k_sum / rank)

    alignment_score = calculate_alignment(model, dataloader, device, idx_to_class, root, classes)

    acc = 100 * correct / total
    cnt = mistake_stats['count'] if mistake_stats['count'] > 0 else 1
    avg_lca_depth_m = mistake_stats['lca_depth'] / cnt
    avg_path_dist = mistake_stats['dist'] / cnt
    avg_dist_to_lca = avg_path_dist / 2.0
    avg_rel_depth_all = rel_depth_all / total
    avg_mistake_rel_depth = mistake_stats['rel_depth'] / cnt

    print("\n--- Evaluation Finished ---")
    print(f"Total Samples: {total}")
    print("-" * 50)
    print(f"Top-1 Accuracy:            {acc:.2f}%")
    print("-" * 50)
    print(f"  Avg LCA Depth (Mistake): {avg_lca_depth_m:.4f}")
    print(f"  Avg Dist to LCA:         {avg_dist_to_lca:.4f}")
    print(f"  Rel. LCA Depth (All):    {avg_rel_depth_all:.4f}")
    print("-" * 50)
    print(f"  Hierarchical Dist (Mistake):  {mistake_height_sum/cnt:.4f}")
    print(f"  Avg Hierarchical Dist @ K=1:  {topk_height_sums[1]/total:.4f}")
    print(f"  Avg Hierarchical Dist @ K=5:  {topk_height_sums[5]/total:.4f}")
    # print(f"  Avg Hierarchical Dist @ K=20: {topk_height_sums[20]/total:.4f}")
    print("-" * 50)
    print(f"Mistake-Only Rel Depth:    {avg_mistake_rel_depth:.4f}")
    print(f"MASTER METRIC (Acc * Mistake): {(acc/100.0) * avg_mistake_rel_depth:.4f}")
    print(f"Tree-Visual Alignment:     {alignment_score:.4f}")
    print("="*50)

    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=list(range(len(classes))))
    
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=False, xticklabels=classes, yticklabels=classes)
    plt.title(f'MBM Classifier Confusion Matrix ({args.split})', fontsize=20)
    plt.ylabel('True Label', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(args.cm_save_path, dpi=300)
    print(f"Confusion matrix saved to {args.cm_save_path}")

    print(f"\n--- Significant Misclassifications (Threshold > {args.misclass_threshold}) ---")
    for true_idx in range(len(classes)):
        for pred_idx in range(len(classes)):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > args.misclass_threshold:
                true_name = classes[true_idx]
                pred_name = classes[pred_idx]
                count = cm[true_idx, pred_idx]
                
                pair_key = (true_name, pred_name)
                metrics = class_pair_mistakes.get(pair_key, [])
                if metrics:
                    lca_d = np.mean([m[0] for m in metrics])
                    avg_d = np.mean([m[1] for m in metrics])
                    print(f"'{true_name}' → '{pred_name}': {count} times")
                    print(f"   • LCA Depth: {lca_d:.3f}")
                    print(f"   • Avg Dist to LCA: {avg_d:.3f}\n")
                else:
                    print(f"'{true_name}' → '{pred_name}': {count} times")
    print("\n---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='./data/fer2013', type=str)
    parser.add_argument('--hierarchy-path', required=True, type=str)
    parser.add_argument('--model-path', required=True, type=str)
    parser.add_argument('--cm-save-path', default='confusion_matrix.png', type=str)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--misclass-threshold', default=20, type=int)
    parser.add_argument('--split', default='val', choices=['val', 'test'], help="Split to evaluate on")
    args = parser.parse_args()
    main(args)