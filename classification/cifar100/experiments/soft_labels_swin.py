import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import timm
import argparse
import json
import numpy as np
import os
import sys
import random
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform, pdist

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import wnid_to_name, download_nltk_data

# ==========================================
# 1. UTILS & TREE HELPERS
# ==========================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.height = 0 # Added to track node height

    def add_child(self, child):
        self.children.append(child)

def build_tree(data, parent=None):
    """Parses the JSON tree."""
    if isinstance(data, str):
        return Node(data.replace(' ', '_'), parent)
    
    node = Node(data.get('name', 'Unnamed').replace(' ', '_'), parent)
    
    if 'children' in data and data['children'] is not None:
        for c in data['children']:
            node.add_child(build_tree(c, node))
            
    return node

def compute_node_heights(node):
    """Recursively computes height (distance to deepest leaf) for every node."""
    if not node.children:
        node.height = 0
        return 0
    
    max_h = 0
    for c in node.children:
        max_h = max(max_h, compute_node_heights(c))
    
    node.height = max_h + 1
    return node.height

def find_node(root, name):
    if root.name == name: return root
    for c in root.children:
        res = find_node(c, name)
        if res: return res
    return None

def get_depth(node):
    d = 0
    while node.parent: 
        d += 1
        node = node.parent
    return d

def find_lca(n1, n2):
    path = set()
    curr = n1
    while curr: 
        path.add(curr)
        curr = curr.parent
    curr = n2
    while curr: 
        if curr in path: return curr
        curr = curr.parent
    return None

def get_leaves(node):
    if not node.children: return [node]
    leaves = []
    for c in node.children: 
        leaves.extend(get_leaves(c))
    return leaves

def calculate_alignment(model, loader, device, idx_to_class, tree_root, classes):
    """
    Calculates the Tree-Visual Alignment (Spearman correlation).
    """
    model.eval()
    class_features = {c: [] for c in classes}
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting Features for Alignment"):
            images = images.to(device)
            # Extract features from the backbone
            features = model.forward_features(images)
            if len(features.shape) > 2: 
                features = features.mean(dim=1)
            
            for i, lbl in enumerate(labels):
                name = idx_to_class[lbl.item()]
                class_features[name].append(features[i].cpu().numpy())

    centroids = []
    valid_classes = []
    for c in classes:
        if class_features[c]:
            centroids.append(np.mean(np.vstack(class_features[c]), axis=0))
            valid_classes.append(c)
            
    if len(centroids) < 2: 
        return 0.0
    
    # 1. Visual Distances (Cosine)
    vis_dists = pdist(np.array(centroids), metric='cosine')
    
    # 2. Tree Distances
    n = len(valid_classes)
    tree_dists = np.zeros((n, n))
    nodes = [find_node(tree_root, c) for c in valid_classes]
    
    for i in range(n):
        for j in range(i+1, n):
            # Tree Dist = depth(u) + depth(v) - 2*depth(lca)
            lca = find_lca(nodes[i], nodes[j])
            lca_depth = get_depth(lca) if lca else 0
            d = get_depth(nodes[i]) + get_depth(nodes[j]) - 2 * lca_depth
            tree_dists[i,j] = tree_dists[j,i] = d
            
    rho, _ = spearmanr(vis_dists, squareform(tree_dists))
    return rho

# ==========================================
# 2. HIERARCHICAL SOFT LOSS
# ==========================================

class HierarchicalSoftLoss(nn.Module):
    def __init__(self, root, classes, beta=1.0, device='cuda'):
        super().__init__()
        self.beta = beta
        self.device = device
        self.num_classes = len(classes)
        
        print(f"Initializing Hierarchical Soft Labels (beta={beta})...")
        
        # Map class indices to Tree Nodes
        self.nodes = []
        for class_name in classes:
            node = find_node(root, class_name)
            if not node:
                print(f"Warning: Class '{class_name}' not found in hierarchy!")
                node = root 
            self.nodes.append(node)
            
        # Pre-compute Distance Matrix (N x N)
        dist_matrix = np.zeros((self.num_classes, self.num_classes))
        print("Pre-computing tree distance matrix...")
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i == j:
                    dist_matrix[i, j] = 0
                else:
                    # Dist(a, b) = depth(a) + depth(b) - 2*depth(lca)
                    n1, n2 = self.nodes[i], self.nodes[j]
                    if n1 and n2:
                        lca = find_lca(n1, n2)
                        lca_depth = get_depth(lca) if lca else 0
                        d = get_depth(n1) + get_depth(n2) - 2 * lca_depth
                        dist_matrix[i, j] = d
                    else:
                        dist_matrix[i, j] = 100 # High penalty for missing nodes
        
        # Convert Distances to Soft Probabilities
        # Formula: P(y=k | target=i) = exp(-beta * dist(i, k)) / Z
        neg_dist = -self.beta * dist_matrix
        # Numerical stability: subtract max per row
        neg_dist = neg_dist - np.max(neg_dist, axis=1, keepdims=True)
        exp_dist = np.exp(neg_dist)
        self.soft_labels = exp_dist / np.sum(exp_dist, axis=1, keepdims=True)
        
        # Convert to Tensor
        self.soft_labels = torch.tensor(self.soft_labels, dtype=torch.float32).to(device)
        print("Soft Label Matrix created.")

    def forward(self, logits, targets):
        """
        logits: (batch, num_classes)
        targets: (batch) - hard indices
        """
        target_dist = self.soft_labels[targets]
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.kl_div(log_probs, target_dist, reduction='batchmean')
        return loss

# ==========================================
# 3. MAIN SCRIPT
# ==========================================

def validate(model, loader, device):
    """Calculates standard Top-1 Accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hierarchy', default='./assets/golden_tree_tiered_imagenet.json', help='Path to hierarchy.json')
    parser.add_argument('--data-path', default='./data/tiered_imagenet_standard', help='Path to dataset root')
    parser.add_argument('--beta', type=float, default=1.0, help='Softness parameter (Lower=Softer)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--save-path', default='soft_labels_tiered_best.pth')
    args = parser.parse_args()
    
    set_seed()
    download_nltk_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading Tiered ImageNet Data from {args.data_path}...")
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_dir = os.path.join(args.data_path, 'train')
    test_dir = os.path.join(args.data_path, 'test')
    
    if not os.path.exists(test_dir):
        val_dir = os.path.join(args.data_path, 'val')
        if os.path.exists(val_dir):
            test_dir = val_dir

    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform_test)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    classes = [wnid_to_name(c).replace(' ', '_') for c in trainset.classes]
    num_classes = len(classes)
    print(f"Detected {num_classes} classes.")

    # 2. Load Hierarchy
    print(f"Loading Hierarchy from {args.hierarchy}...")
    with open(args.hierarchy) as f: 
        tree_data = json.load(f)
    root = build_tree(tree_data)
    compute_node_heights(root)
    
    # 3. Model & Loss
    print(f"Initializing Swin with Soft Labels (Beta={args.beta})...")
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
    model = model.to(device)
    
    criterion = HierarchicalSoftLoss(root, classes, beta=args.beta, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    # 4. Training Loop
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        val_acc = validate(model, testloader, device)
        avg_loss = total_loss / len(trainloader)
        print(f"Epoch {epoch+1} Summary - Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            print(f"--> Accuracy Improved ({best_acc:.2f}% -> {val_acc:.2f}%). Saving model...")
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            
    print(f"\nTraining Finished. Best Validation Accuracy: {best_acc:.2f}%")

    # 5. Full Hierarchical Evaluation
    print("\nLoading best model for detailed evaluation...")
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    
    correct = 0
    total = 0
    mistake_stats = {'lca_depth': 0, 'rel_depth': 0, 'dist': 0, 'count': 0}
    rel_depth_all = 0.0 # Track for ALL samples
    
    mistake_height_sum = 0.0
    topk_height_sums = {1: 0.0, 5: 0.0, 20: 0.0}

    idx_to_class = {i: c for i,c in enumerate(classes)}
    leaves = get_leaves(root)
    valid_leaves = [l for l in leaves if l.name in classes]
    avg_leaf_depth = sum(get_depth(l) for l in valid_leaves) / len(valid_leaves) if valid_leaves else 1.0

    print("Running Inference & Metrics...")
    with torch.no_grad():
        for imgs, lbls in tqdm(testloader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            _, topk_preds = logits.topk(20, dim=1)
            
            for i in range(len(lbls)):
                total += 1
                t_idx = lbls[i].item()
                p_idx = topk_preds[i, 0].item()
                
                node_t = find_node(root, idx_to_class[t_idx])
                node_p = find_node(root, idx_to_class[p_idx])
                
                if node_t and node_p:
                    lca = find_lca(node_t, node_p)
                    lca_d = get_depth(lca) if lca else 0
                    rel_d = lca_d / avg_leaf_depth
                    
                    # Accumulate for ALL samples (Correct + Incorrect)
                    rel_depth_all += rel_d
                    
                    if t_idx == p_idx:
                        correct += 1
                    else:
                        mistake_stats['count'] += 1
                        mistake_stats['lca_depth'] += lca_d
                        mistake_stats['rel_depth'] += rel_d # Accumulate for mistakes only
                        
                        d_t = get_depth(node_t)
                        d_p = get_depth(node_p)
                        mistake_stats['dist'] += (d_t + d_p - 2*lca_d) # Full path distance

                        mistake_height_sum += lca.height if lca else 0
                
                # Top-K Metrics
                if node_t:
                    curr_sample_k_sum = 0.0
                    for k in range(20):
                        curr_p_idx = topk_preds[i, k].item()
                        node_curr_p = find_node(root, idx_to_class[curr_p_idx])
                        if node_curr_p:
                            curr_lca = find_lca(node_t, node_curr_p)
                            if curr_lca:
                                curr_sample_k_sum += curr_lca.height
                        
                        rank = k + 1
                        if rank in topk_height_sums:
                            topk_height_sums[rank] += (curr_sample_k_sum / rank)

    print("Calculating Tree-Visual Alignment...")
    align_score = calculate_alignment(model, testloader, device, idx_to_class, root, classes)
    
    # Statistics calculation
    cnt = mistake_stats['count'] if mistake_stats['count'] > 0 else 1
    avg_path_dist = mistake_stats['dist'] / cnt
    avg_dist_to_lca = avg_path_dist / 2.0  # <--- Fix 2: Halve the path distance

    print("\n" + "="*50)
    print("SOFT LABELS METHODOLOGY RESULTS")
    print("="*50)
    print(f"Top-1 Accuracy:            {100*correct/total:.2f}%")
    print("-" * 50)
    print("Existing Metrics:")
    print(f"  Avg LCA Depth (Mistake): {mistake_stats['lca_depth']/cnt:.4f}")
    # Fix 2 Reported Here
    print(f"  Avg Dist to LCA:         {avg_dist_to_lca:.4f}") 
    # Fix 1 Reported Here: Rel. LCA Depth uses 'total', not 'cnt'
    print(f"  Rel. LCA Depth (All):    {rel_depth_all/total:.4f}")
    print("-" * 50)
    print("New Metrics:")
    print(f"  Hierarchical Dist (Mistake):  {mistake_height_sum/cnt:.4f}")
    print(f"  Avg Hierarchical Dist @ K=1:  {topk_height_sums[1]/total:.4f}")
    print(f"  Avg Hierarchical Dist @ K=5:  {topk_height_sums[5]/total:.4f}")
    print(f"  Avg Hierarchical Dist @ K=20: {topk_height_sums[20]/total:.4f}")
    print("-" * 50)
    # Fix 1 Part 2: Mistake Only Rel Depth remains mistake only
    print(f"Mistake-Only Rel Depth:    {mistake_stats['rel_depth']/cnt:.4f}")
    print(f"Tree-Visual Alignment:     {align_score:.4f}")
    print("="*50)

if __name__ == '__main__':
    main()