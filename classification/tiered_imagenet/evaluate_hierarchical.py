import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils import build_tree, wnid_to_name, download_nltk_data
import argparse
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import timm 
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

# --- Helper Functions ---

def get_unique_node_name(node):
    """Generates a unique name for the node to avoid collisions."""
    if node.parent:
        return f"{node.name}_via_{node.parent.name}"
    return node.name

def get_paths_to_root(node):
    """Generates a list of nodes from the given node up to the root."""
    path = []
    curr = node
    while curr:
        path.append(curr)
        curr = curr.parent
    return path

def find_lca(node1, node2):
    """Finds the Lowest Common Ancestor of two nodes in the tree."""
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
    """Recursively gets all leaf nodes under a given node."""
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

def calculate_mistake_severity_metrics(true_node, pred_node, avg_leaf_depth):
    if not true_node or not pred_node:
        return None, None, None

    lca = find_lca(true_node, pred_node)
    if not lca:
        return None, None, None

    # Metric 1: LCA depth
    lca_depth = get_depth(lca)

    # Metric 2: Average distance from nodes to LCA
    dist_true_to_lca = 0
    curr = true_node
    while curr != lca:
        dist_true_to_lca += 1
        curr = curr.parent

    dist_pred_to_lca = 0
    curr = pred_node
    while curr != lca:
        dist_pred_to_lca += 1
        curr = curr.parent

    avg_dist_to_lca = (dist_true_to_lca + dist_pred_to_lca) / 2.0
    
    # Metric 3: Relative LCA Depth
    relative_lca_depth = lca_depth / avg_leaf_depth if avg_leaf_depth > 0 else 0

    return lca_depth, avg_dist_to_lca, relative_lca_depth

def find_node_in_tree(node, name):
    """Recursively finds a node by its name in the tree."""
    if node.name == name:
        return node
    for child in node.children:
        found = find_node_in_tree(child, name)
        if found:
            return found
    return None

def load_all_classifiers(node, device, num_ftrs, weights_dir):
    """Recursively loads the trained .pth files for each internal node."""
    if not node.children:
        return

    num_children = len(node.children)
    classifier = nn.Linear(num_ftrs, num_children)
    
    # USE UNIQUE NAME HERE TO MATCH TRAIN SCRIPT
    unique_name = get_unique_node_name(node)
    model_path = os.path.join(weights_dir, f"{unique_name}_classifier.pth")
    
    if os.path.exists(model_path):
        try:
            classifier.load_state_dict(torch.load(model_path, map_location=device))
            classifier.to(device)
            classifier.eval()
            node.classifier = classifier
        except RuntimeError as e:
            print(f"Error loading {unique_name}: {e}")
            node.classifier = None
    else:
        # print(f"Warning: Classifier weights not found for node {unique_name}")
        node.classifier = None

    for child in node.children:
        load_all_classifiers(child, device, num_ftrs, weights_dir)

# --- Inference Function: Beam Search ---
def predict_leaf_beam_search(features, root_node, beam_width=3):
    candidates = [(root_node, 0.0)]
    final_leaves = []
    
    while candidates:
        next_candidates = []
        
        for node, current_score in candidates:
            if not node.children:
                final_leaves.append((node, current_score))
                continue
            
            if not node.classifier:
                final_leaves.append((node, current_score))
                continue
                
            with torch.no_grad():
                logits = node.classifier(features)
                log_probs = F.log_softmax(logits, dim=0)
            
            k = len(node.children) 
            top_scores, top_indices = torch.topk(log_probs, k)
            
            for score, idx in zip(top_scores, top_indices):
                child = node.children[idx.item()]
                new_score = current_score + score.item()
                next_candidates.append((child, new_score))
        
        if next_candidates:
            next_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = next_candidates[:beam_width]
        else:
            break
            
    if not final_leaves:
        return "prediction_error"

    best_leaf, _ = max(final_leaves, key=lambda x: x[1])
    return best_leaf.name

# --- Metric Function: Tree-Visual Alignment ---
def calculate_tree_confusion_alignment(cm, class_names, root):
    n = len(class_names)
    tree_dist_matrix = np.zeros((n, n))
    nodes = {name: find_node_in_tree(root, name) for name in class_names}
    
    for i in range(n):
        for j in range(i + 1, n):
            node_i = nodes.get(class_names[i])
            node_j = nodes.get(class_names[j])
            
            if node_i and node_j:
                lca = find_lca(node_i, node_j)
                if lca:
                    dist = get_depth(node_i) + get_depth(node_j) - 2 * get_depth(lca)
                    tree_dist_matrix[i, j] = dist
                    tree_dist_matrix[j, i] = dist
                else:
                    tree_dist_matrix[i, j] = 100 
                    tree_dist_matrix[j, i] = 100
            else:
                tree_dist_matrix[i, j] = 100 
                tree_dist_matrix[j, i] = 100

    tri_idx = np.triu_indices(n, k=1)
    vec_tree_dist = tree_dist_matrix[tri_idx]
    vec_confusion = cm[tri_idx]
    corr, _ = spearmanr(vec_confusion, vec_tree_dist)
    return corr

def main(args):
    print("Starting hierarchical evaluation (Tiered ImageNet)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    download_nltk_data()

    # 1. Load Hierarchy
    with open(args.hierarchy_path, 'r') as f:
        hierarchy_data = json.load(f)
    root = build_tree(hierarchy_data)
    avg_leaf_depth = get_average_leaf_depth(root)
    print(f"Hierarchy tree built. Average leaf depth: {avg_leaf_depth:.4f}")

    # 2. Load Data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    test_dir = os.path.join(args.data_path, 'test')
    test_dataset = ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    idx_to_class = {} 
    class_names = []  
    
    for i, wnid in enumerate(test_dataset.classes):
        human_name = wnid_to_name(wnid).replace(' ', '_')
        idx_to_class[i] = human_name
        class_names.append(human_name)

    class_to_idx_tree_format = {v: k for k, v in idx_to_class.items()}
    print(f"Tiered ImageNet test dataset loaded. Found {len(class_names)} classes.")

    # 3. Load Backbone
    print("Loading Swin Transformer backbone...")
    backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0) 
    num_ftrs = backbone.num_features 
    
    state_dict = torch.load(args.backbone_path, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
    backbone.load_state_dict(state_dict, strict=False)
    backbone.to(device)
    backbone.eval()
    print("Swin backbone loaded and in evaluation mode.")

    # 4. Load Classifiers
    load_all_classifiers(root, device, num_ftrs, args.weights_dir)
    print("All node classifiers loaded into the tree.")

    # --- EVALUATION LOOP ---
    total_samples = 0
    correct_predictions = 0
    total_lca_depth = 0
    total_avg_dist_to_lca = 0
    total_relative_lca_depth = 0
    total_relative_lca_depth_mistakes_only = 0.0
    
    all_true_labels = []
    all_pred_labels = []
    class_pair_mistakes = {}

    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        with torch.no_grad():
            features_batch = backbone(images)

        for i in range(images.size(0)):
            features_single = features_batch[i]
            true_label_idx = labels[i].item()

            predicted_leaf_name = predict_leaf_beam_search(features_single, root, beam_width=5)
            
            true_leaf_name = idx_to_class[true_label_idx]

            node1 = find_node_in_tree(root, true_leaf_name)
            node2 = find_node_in_tree(root, predicted_leaf_name)

            if predicted_leaf_name == true_leaf_name:
                correct_predictions += 1
                total_relative_lca_depth += 1.0 
            else:
                if node1 and node2:
                    lca_depth, avg_dist_to_lca, relative_lca_depth = calculate_mistake_severity_metrics(node1, node2, avg_leaf_depth)
                    
                    if lca_depth is not None:
                        total_lca_depth += lca_depth
                        total_avg_dist_to_lca += avg_dist_to_lca
                        total_relative_lca_depth += relative_lca_depth
                        total_relative_lca_depth_mistakes_only += relative_lca_depth
                        
                        pair_key = (true_leaf_name, predicted_leaf_name)
                        if pair_key not in class_pair_mistakes:
                            class_pair_mistakes[pair_key] = []
                        class_pair_mistakes[pair_key].append((lca_depth, avg_dist_to_lca, relative_lca_depth))

            total_samples += 1
            predicted_label_idx = class_to_idx_tree_format.get(predicted_leaf_name, -1)
            all_true_labels.append(true_label_idx)
            all_pred_labels.append(predicted_label_idx)

    # --- METRICS CALCULATION ---
    accuracy = (correct_predictions / total_samples) * 100
    num_mistakes = total_samples - correct_predictions
    
    avg_lca_depth = total_lca_depth / total_samples 
    avg_dist_to_lca_overall = total_avg_dist_to_lca / total_samples
    avg_relative_lca_depth = total_relative_lca_depth / total_samples

    if num_mistakes > 0:
        avg_relative_lca_depth_mistakes = total_relative_lca_depth_mistakes_only / num_mistakes
    else:
        avg_relative_lca_depth_mistakes = 1.0

    cm_labels = list(range(len(class_names)))
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=cm_labels)
    alignment_score = calculate_tree_confusion_alignment(cm, class_names, root)

    print("\n--- Evaluation Finished ---")
    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print("-" * 40)
    print(f"Top-1 Accuracy: {accuracy:.2f}%")
    print(f"Mistake Severity Metrics:")
    print(f"   • LCA Depth (Metric 1): {avg_lca_depth:.4f}")
    print(f"   • Avg Dist to LCA (Metric 2): {avg_dist_to_lca_overall:.4f}")
    print(f"   • Relative LCA Depth (Metric 3): {avg_relative_lca_depth:.4f}")
    print("-" * 40)
    print(f"--- NEW Optimization Metrics ---")
    print(f"   • Mistake-Only Relative Depth: {avg_relative_lca_depth_mistakes:.4f} (Higher is Better)")
    print(f"   • Tree-Visual Alignment (Spearman): {alignment_score:.4f} (More Negative is Better)")
    print("-" * 40)
    
    print("\n--- Confusion Matrix ---")
    print("\nGenerating and saving confusion matrix heatmap...")
    plt.figure(figsize=(40, 36)) 
    sns.heatmap(cm, annot=False, xticklabels=False, yticklabels=False)
    plt.title('Hierarchical Classifier Confusion Matrix', fontsize=20)
    plt.ylabel('True Label', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.tight_layout()
    plt.savefig(args.cm_save_path, dpi=300)
    print(f"Confusion matrix heatmap saved to {args.cm_save_path}")
    
    print(f"\n--- Significant Misclassifications (Threshold > {args.misclass_threshold}) ---")
    
    found_misclass = False
    for true_idx in range(len(class_names)):
        for pred_idx in range(len(class_names)):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > args.misclass_threshold:
                true_name = class_names[true_idx]
                pred_name = class_names[pred_idx]
                count = cm[true_idx, pred_idx]
                
                pair_key = (true_name, pred_name)
                if pair_key in class_pair_mistakes:
                    mistakes_data = class_pair_mistakes[pair_key]
                    avg_lca_depth_pair = np.mean([m[0] for m in mistakes_data])
                    avg_dist_to_lca_pair = np.mean([m[1] for m in mistakes_data])
                    
                    print(f"'{true_name}' → '{pred_name}': {count} times")
                    print(f"   • LCA Depth: {avg_lca_depth_pair:.3f}")
                    print(f"   • Avg Dist to LCA: {avg_dist_to_lca_pair:.3f}\n")
                else:
                    print(f"'{true_name}' → '{pred_name}': {count} times")
                    print("   • Severity metrics: Not available\n")
                found_misclass = True

    if not found_misclass:
        print("No significant misclassifications found above the specified threshold.")
    print("\n---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a hierarchical classifier on Tiered ImageNet.')
    parser.add_argument('--data-path', default='./data/tiered_imagenet_standard', type=str, help='Path to dataset directory (contains test folder).')
    parser.add_argument('--hierarchy-path', required=True, type=str, help='Path to the JSON hierarchy file.')
    parser.add_argument('--weights-dir', default='./weights', type=str, help='Directory containing all classifier .pth files.')
    parser.add_argument('--backbone-path', required=True, type=str, help='Path to the frozen backbone weights.')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size for evaluation.')
    parser.add_argument('--cm-save-path', default='confusion_matrix.png', type=str, help='Path to save the confusion matrix image.')
    parser.add_argument('--misclass-threshold', default=20, type=int, help='Threshold for printing significant misclassifications.')
    
    args = parser.parse_args()
    main(args)