# evaluate_flat_model.py
# Uses the EXACT relative depth metrics (1.0 = best) from your eval scripts.

import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import timm
import argparse
import os

# --- Helper Class: Node (copied from utils.py) ---
class Node:
    """A node in the hierarchy tree."""
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
    def add_child(self, child_node):
        self.children.append(child_node)

def build_tree(data, parent=None):
    """Recursively builds the hierarchy tree from the JSON data."""
    if isinstance(data, str):
        node_name = data.replace(' ', '_')
        node = Node(node_name, parent)
        return node
    node_name = data.get('name', 'Unnamed Node').replace(' ', '_')
    node = Node(node_name, parent)
    if 'children' in data and data['children'] is not None:
        for child_data in data['children']:
            if child_data:
                child_node = build_tree(child_data, node)
                node.add_child(child_node)
    return node

# --- START: Helpers from evaluate_hierarchical.py (for curr_tree) ---
def get_paths_to_root(node):
    path = []
    curr = node
    while curr:
        path.append(curr)
        curr = curr.parent
    return path

def find_lca(node1, node2):
    path1 = get_paths_to_root(node1)
    path2 = get_paths_to_root(node2)
    path2_nodes = set(path2)
    for node in path1:
        if node in path2_nodes:
            return node
    return None

def get_depth(node):
    """Calculates the depth of a node (root is at depth 0)."""
    depth = 0
    curr = node
    while curr.parent:
        depth += 1
        curr = curr.parent
    return depth

def get_all_leaf_nodes(node, leaves):
    if not node.children:
        leaves.append(node)
        return
    for child in node.children:
        get_all_leaf_nodes(child, leaves)

def get_average_leaf_depth(root_node):
    """Calculates the average depth of all leaf nodes in the tree."""
    leaf_nodes = []
    get_all_leaf_nodes(root_node, leaf_nodes)
    if not leaf_nodes:
        return 0
    total_depth = sum(get_depth(leaf) for leaf in leaf_nodes)
    return total_depth / len(leaf_nodes)

def find_node_in_tree(node, name):
    if node.name == name:
        return node
    for child in node.children:
        found = find_node_in_tree(child, name)
        if found:
            return found
    return None
# --- END: Helpers from evaluate_hierarchical.py ---


# --- START: Helpers from eval_haf.py (for haf_tree) ---
trees = [
	[0, 4, 5, 3, 0], [1, 1, 6, 0, 0], [2, 14, 3, 0, 0], [3, 8, 3, 0, 0],
    [4, 0, 6, 0, 0], [5, 6, 0, 1, 1], [6, 7, 4, 0, 0], [7, 7, 4, 0, 0],
    [8, 18, 7, 1, 1], [9, 3, 0, 1, 1], [10, 3, 0, 1, 1], [11, 14, 3, 0, 0],
    [12, 9, 1, 1, 1], [13, 18, 7, 1, 1], [14, 7, 4, 0, 0], [15, 11, 3, 0, 0],
    [16, 3, 0, 1, 1], [17, 9, 1, 1, 1], [18, 7, 4, 0, 0], [19, 11, 3, 0, 0],
    [20, 6, 0, 1, 1], [21, 11, 3, 0, 0], [22, 5, 0, 1, 1], [23, 10, 2, 2, 1],
    [24, 7, 4, 0, 0], [25, 6, 0, 1, 1], [26, 13, 4, 0, 0], [27, 15, 4, 0, 0],
    [28, 3, 0, 1, 1], [29, 15, 4, 0, 0], [30, 0, 6, 0, 0], [31, 11, 3, 0, 0],
    [32, 1, 6, 0, 0], [33, 10, 2, 2, 1], [34, 12, 3, 0, 0], [35, 14, 3, 0, 0],
    [36, 16, 3, 0, 0], [37, 9, 1, 1, 1], [38, 11, 3, 0, 0], [39, 5, 0, 1, 1],
    [40, 5, 0, 1, 1], [41, 19, 7, 1, 1], [42, 8, 3, 0, 0], [43, 8, 3, 0, 0],
    [44, 15, 4, 0, 0], [45, 13, 4, 0, 0], [46, 14, 3, 0, 0], [47, 17, 5, 3, 0],
    [48, 18, 7, 1, 1], [49, 10, 2, 2, 1], [50, 16, 3, 0, 0], [51, 4, 5, 3, 0],
    [52, 17, 5, 3, 0], [53, 4, 5, 3, 0], [54, 2, 5, 3, 0], [55, 0, 6, 0, 0],
    [56, 17, 5, 3, 0], [57, 4, 5, 3, 0], [58, 18, 7, 1, 1], [59, 17, 5, 3, 0],
    [60, 10, 2, 2, 1], [61, 3, 0, 1, 1], [62, 2, 5, 3, 0], [63, 12, 3, 0, 0],
    [64, 12, 3, 0, 0], [65, 16, 3, 0, 0], [66, 12, 3, 0, 0], [67, 1, 6, 0, 0],
    [68, 9, 1, 1, 1], [69, 19, 7, 1, 1], [70, 2, 5, 3, 0], [71, 10, 2, 2, 1],
    [72, 0, 6, 0, 0], [73, 1, 6, 0, 0], [74, 16, 3, 0, 0], [75, 12, 3, 0, 0],
    [76, 9, 1, 1, 1], [77, 13, 4, 0, 0], [78, 15, 4, 0, 0], [79, 13, 4, 0, 0],
    [80, 16, 3, 0, 0], [81, 19, 7, 1, 1], [82, 2, 5, 3, 0], [83, 4, 5, 3, 0],
    [84, 6, 0, 1, 1], [85, 19, 7, 1, 1], [86, 5, 0, 1, 1], [87, 5, 0, 1, 1],
    [88, 8, 3, 0, 0], [89, 19, 7, 1, 1], [90, 18, 7, 1, 1], [91, 1, 6, 0, 0],
    [92, 2, 5, 3, 0], [93, 15, 4, 0, 0], [94, 6, 0, 1, 1], [95, 0, 6, 0, 0],
    [96, 17, 5, 3, 0], [97, 8, 3, 0, 0], [98, 14, 3, 0, 0], [99, 13, 4, 0, 0]
]

def get_lca_level(class1_idx, class2_idx):
    """
    Finds the DEPTH (0-4) of the LCA for the HAF tree.
    Depth 0 = root, Depth 4 = leaf.
    Returns -1 if paths are identical (correct prediction).
    """
    path1 = trees[class1_idx]
    path2 = trees[class2_idx]
    
    if path1 == path2:
        return -1 # Signal for identical paths

    lca_depth = 0 # Default to root (depth 0)
    
    # Iterate from root (idx 4, depth 0) downwards to leaf (idx 0, depth 4)
    # The loop range(start, stop, step) is [4, 3, 2, 1, 0]
    for level_idx in range(len(path1) - 1, -1, -1):
        if path1[level_idx] == path2[level_idx]:
            # This node is common. Calculate its depth.
            # Depth = (total_indices_5 - 1) - level_index
            current_depth = (len(path1) - 1) - level_idx
            lca_depth = current_depth
        else:
            # Paths have diverged. The 'lca_depth' we just saved is the correct one.
            return lca_depth
    
    # This case is for when one path is a subset of another (not in CIFAR-100)
    # or they are identical (caught above).
    return lca_depth

def calculate_haf_relative_depth(class1_idx, class2_idx, avg_leaf_depth):
    """
    This is the exact metric from eval_haf.py.
    """
    lca_level = get_lca_level(class1_idx, class2_idx)
    
    if lca_level == -1:
        # This means the paths are identical (a correct prediction)
        # eval_haf.py explicitly returns 1.0 in this case.
        return 1.0
        
    lca_depth = lca_level
    if avg_leaf_depth == 0:
        return 0
    return lca_depth / avg_leaf_depth
# --- END: Helpers from eval_haf.py ---


# --- Helper function to process and save rankings ---
def process_and_save_rankings(severity_data, save_path):
    """
    Aggregates relative depth scores, sorts them (descending), and saves.
    """
    avg_scores = {}
    for (true_class, pred_class), scores in severity_data.items():
        if not scores:
            continue
        avg_score = np.mean(scores)
        if true_class not in avg_scores:
            avg_scores[true_class] = []
        avg_scores[true_class].append(
            {"misclassified_as": pred_class, "relative_depth_score": avg_score}
        )
    
    for true_class in avg_scores:
        # Sort by relative_depth_score (descending). 1.0 = best.
        avg_scores[true_class].sort(key=lambda x: x['relative_depth_score'], reverse=True)
    
    with open(save_path, 'w') as f:
        json.dump(avg_scores, f, indent=2)
    print(f"Misclassification ranking (1.0=best) saved to {save_path}")

# --- Main Evaluation Function ---
def main(args):
    print("--- Evaluating Flat Swin-T Model with Dual-Tree Relative Depth ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Trees and Get Depth Metrics ---
    
    # Load curr_tree.json (from evaluate_hierarchical.py logic)
    with open(args.curr_tree_path, 'r') as f:
        curr_tree_data = json.load(f)
    curr_tree_root = build_tree(curr_tree_data)
    avg_leaf_depth_curr_tree = get_average_leaf_depth(curr_tree_root)
    print(f"Loaded 'curr_tree.json'. Avg. leaf depth: {avg_leaf_depth_curr_tree:.4f}")

    # Load tree_haf.json (from eval_haf.py logic)
    avg_leaf_depth_baseline_haf = 4.0 # This is the hardcoded value from eval_haf.py
    print(f"Loaded 'tree_haf.json' logic. Fixed avg. leaf depth: {avg_leaf_depth_baseline_haf}")

    # --- 2. Load Dataset ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    test_dataset = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    idx_to_class_name = {i: c.replace(' ', '_') for i, c in enumerate(test_dataset.classes)}
    print("CIFAR-100 test dataset loaded.")

    # --- 3. Load FULL Swin-T Model ---
    print("Loading FULL Swin Transformer model...")
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=100)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Full Swin-T model loaded from {args.model_path}.")

    # --- 4. Run Evaluation and Collect Scores ---
    
    # Dictionaries for ranking
    curr_tree_scores = {}
    haf_tree_scores = {}

    # Accumulators for the two overall metrics
    total_relative_depth_curr_tree = 0.0
    total_relative_depth_haf_tree = 0.0
    
    total_samples = 0
    correct_predictions = 0

    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            _, predicted_indices = torch.max(outputs, 1)

        for i in range(images.size(0)):
            true_label_idx = labels[i].item()
            predicted_label_idx = predicted_indices[i].item()
            total_samples += 1
            
            # --- Check if prediction is correct ---
            if predicted_label_idx == true_label_idx:
                correct_predictions += 1
                # A correct prediction gets a score of 1.0 for both metrics
                total_relative_depth_curr_tree += 1.0
                total_relative_depth_haf_tree += 1.0
                continue
                
            # --- This is a misclassification, calculate both scores ---
            
            true_leaf_name = idx_to_class_name[true_label_idx]
            predicted_leaf_name = idx_to_class_name[predicted_label_idx]
            pair_key = (true_leaf_name, predicted_leaf_name)

            # Metric 1: HAF Tree Score (from eval_haf.py)
            haf_score = calculate_haf_relative_depth(true_label_idx, predicted_label_idx, avg_leaf_depth_baseline_haf)
            total_relative_depth_haf_tree += haf_score
            if pair_key not in haf_tree_scores: haf_tree_scores[pair_key] = []
            haf_tree_scores[pair_key].append(haf_score)
            
            # Metric 2: Your Tree Score (from evaluate_hierarchical.py)
            node1 = find_node_in_tree(curr_tree_root, true_leaf_name)
            node2 = find_node_in_tree(curr_tree_root, predicted_leaf_name)
            
            curr_score = 0.0 # Default to 0 if nodes aren't found
            if node1 and node2 and avg_leaf_depth_curr_tree > 0:
                lca = find_lca(node1, node2)
                if lca:
                    lca_depth = get_depth(lca)
                    # This is Metric 3 from your script
                    curr_score = lca_depth / avg_leaf_depth_curr_tree 
            
            total_relative_depth_curr_tree += curr_score
            if pair_key not in curr_tree_scores: curr_tree_scores[pair_key] = []
            curr_tree_scores[pair_key].append(curr_score)

    # --- 5. Report Final Stats and Save Rankings ---
    
    accuracy = (correct_predictions / total_samples) * 100
    
    # Calculate the two final average metrics
    avg_rel_depth_curr = total_relative_depth_curr_tree / total_samples
    avg_rel_depth_haf = total_relative_depth_haf_tree / total_samples

    print("\n--- Evaluation Finished ---")
    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Top-1 Accuracy: {accuracy:.2f}%")
    print("-" * 40)
    print("--- Overall Relative Depth Scores (1.0 = Best) ---")
    print(f"Average Relative Depth (curr_tree.json):   {avg_rel_depth_curr:.4f}")
    print(f"Average Relative Depth (tree_haf.json):     {avg_rel_depth_haf:.4f}")
    print("-" * 40)

    # Process and save the two ranking files
    print("Processing and saving 'curr_tree' severity rankings...")
    process_and_save_rankings(curr_tree_scores, args.curr_tree_ranking_path)
    
    print("Processing and saving 'haf_tree' severity rankings...")
    process_and_save_rankings(haf_tree_scores, args.haf_tree_ranking_path)
    
    print("\nDone.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Swin-T flat classifier and rank misclassifications.')
    
    parser.add_argument('--data-path', default='./data', type=str, 
                        help='Path to CIFAR-100 dataset directory.')
    parser.add_argument('--model-path', default='cifar100_swin_tiny_full_model.pth', type=str, 
                        help='Path to the trained FULL Swin-T model weights (.pth file).')
    parser.add_argument('--curr-tree-path', default='assets/curr_tree.json', type=str, 
                        help='Path to the JSON hierarchy file (curr_tree.json) for severity analysis.')
    
    parser.add_argument('--batch-size', default=64, type=int, 
                        help='Batch size for evaluation.')
    
    parser.add_argument('--curr-tree-ranking-path', default='rankings_flat_curr_tree.json', type=str, 
                        help='Output path for the misclassification ranking based on curr_tree.')
    parser.add_argument('--haf-tree-ranking-path', default='rankings_flat_haf_tree.json', type=str, 
                        help='Output path for the misclassification ranking based on haf_tree.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Full model weights not found at {args.model_path}")
        print("Please run train_swin_flat.py first or provide the correct path.")
    elif not os.path.exists(args.curr_tree_path):
        print(f"Error: Hierarchy file not found at {args.curr_tree_path}")
    else:
        main(args)