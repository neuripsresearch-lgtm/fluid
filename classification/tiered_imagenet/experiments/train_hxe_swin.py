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
    def add_child(self, child):
        self.children.append(child)

def build_tree(data, parent=None):
    """
    Parses the JSON tree. 
    Handles "children": null in the JSON structure.
    """
    if isinstance(data, str):
        return Node(data.replace(' ', '_'), parent)
    
    node = Node(data.get('name', 'Unnamed').replace(' ', '_'), parent)
    
    if 'children' in data and data['children'] is not None:
        for c in data['children']:
            node.add_child(build_tree(c, node))
            
    return node

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

def get_path_to_root(node):
    path = []
    curr = node
    while curr: 
        path.append(curr)
        curr = curr.parent
    return path

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
# 2. HIERARCHICAL CROSS ENTROPY LOSS
# ==========================================

class HierarchicalCrossEntropyLoss(nn.Module):
    def __init__(self, root, classes, alpha=0.5, device='cuda'):
        super().__init__()
        self.alpha = alpha
        self.device = device
        
        # Pre-compute paths and weights for every class index
        self.nodes = [find_node(root, c) for c in classes]
        self.paths = []
        self.weights = []
        self.leaf_indices = []
        
        for i, node in enumerate(self.nodes):
            if not node:
                print(f"Warning: Class '{classes[i]}' not found in hierarchy tree!")
                self.paths.append([])
                self.weights.append([])
                self.leaf_indices.append([])
                continue
                
            path = get_path_to_root(node)[:-1] # Exclude root
            self.paths.append(path)
            # Weight = exp(-alpha * depth)
            self.weights.append([np.exp(-alpha * get_depth(n)) for n in path])
            
            # For HXE, we need to know which class indices fall under which node in the path
            path_indices = []
            for p_node in path:
                leaves_under = get_leaves(p_node)
                # Find the indices (0-N) corresponding to these leaf names
                # Creating a set for O(1) lookup
                leaf_names = set(l.name for l in leaves_under)
                indices = [idx for idx, name in enumerate(classes) if name in leaf_names]
                path_indices.append(indices)
            self.leaf_indices.append(path_indices)

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        batch_size = logits.size(0)
        
        # Standard Cross Entropy part (optional but helps stability)
        ce_loss = F.cross_entropy(logits, targets)
        
        hxe_loss = 0.0
        for b in range(batch_size):
            t = targets[b].item()
            path_w = self.weights[t]
            path_idxs = self.leaf_indices[t]
            
            # Iterate down the path (from leaf up to root's child)
            for l in range(len(path_w) - 1):
                # Conditional Probability: P(Child | Parent) = P(Child_Subtree) / P(Parent_Subtree)
                child_idxs = path_idxs[l]
                parent_idxs = path_idxs[l+1]
                
                p_child = probs[b, child_idxs].sum()
                p_parent = probs[b, parent_idxs].sum()
                
                cond_prob = p_child / (p_parent + 1e-9)
                
                # Weighted log likelihood
                hxe_loss -= path_w[l] * torch.log(cond_prob + 1e-9)
                
        return ce_loss + (hxe_loss / batch_size)

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
    parser.add_argument('--data-path', default='./data/tiered_imagenet_standard', help='Path to dataset root (containing train/test)')
    parser.add_argument('--alpha', type=float, default=0.5, help='HXE Alpha parameter')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--save-path', default='hxe_tiered_best_model.pth')
    args = parser.parse_args()
    
    set_seed()
    download_nltk_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading Tiered ImageNet Data from {args.data_path}...")
    
    # ImageNet Standard Normalization
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

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        # Fallback to 'val' if 'test' doesn't exist, as per some configs
        val_dir = os.path.join(args.data_path, 'val')
        if os.path.exists(val_dir):
            print("Using 'val' directory as test/validation set.")
            test_dir = val_dir
        else:
            raise FileNotFoundError(f"Neither 'test' nor 'val' directory found in {args.data_path}")

    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform_test)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Map WNIDs (folder names) to Human Readable Names
    # trainset.classes are sorted WNIDs. We convert them to names for the hierarchy.
    classes = [wnid_to_name(c).replace(' ', '_') for c in trainset.classes]
    num_classes = len(classes)
    print(f"Detected {num_classes} classes. Examples: {classes[:5]}")
    
    # 2. Load Hierarchy
    print(f"Loading Hierarchy from {args.hierarchy}...")
    with open(args.hierarchy) as f: 
        tree_data = json.load(f)
    root = build_tree(tree_data)
    
    # 3. Model & Loss
    print(f"Initializing HXE Model (Alpha={args.alpha}) for {num_classes} classes...")
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
    model = model.to(device)
    
    criterion = HierarchicalCrossEntropyLoss(root, classes, alpha=args.alpha, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    
    # 4. Training Loop with Validation
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
            
        # Validation
        val_acc = validate(model, testloader, device)
        avg_loss = total_loss / len(trainloader)
        print(f"Epoch {epoch+1} Summary - Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save Best
        if val_acc > best_acc:
            print(f"--> Accuracy Improved ({best_acc:.2f}% -> {val_acc:.2f}%). Saving model...")
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            
    print(f"\nTraining Finished. Best Validation Accuracy: {best_acc:.2f}%")

    # 5. Full Hierarchical Evaluation (using Best Model)
    print("\nLoading best model for detailed evaluation...")
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    
    correct = 0
    total = 0
    mistake_stats = {'lca_depth': 0, 'rel_depth': 0, 'dist': 0, 'count': 0}
    rel_depth_all = 0
    
    idx_to_class = {i: c for i,c in enumerate(classes)}
    leaves = get_leaves(root)
    # Filter out empty/unnamed leaves if any
    valid_leaves = [l for l in leaves if l.name in classes]
    avg_leaf_depth = sum(get_depth(l) for l in valid_leaves) / len(valid_leaves) if valid_leaves else 1.0

    print("Running Inference & Metrics...")
    with torch.no_grad():
        for imgs, lbls in tqdm(testloader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(1)
            
            for i in range(len(lbls)):
                total += 1
                t_idx, p_idx = lbls[i].item(), preds[i].item()
                
                node_t = find_node(root, idx_to_class[t_idx])
                node_p = find_node(root, idx_to_class[p_idx])
                
                if node_t and node_p:
                    lca = find_lca(node_t, node_p)
                    lca_d = get_depth(lca) if lca else 0
                    rel_d = lca_d / avg_leaf_depth
                    
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

    # 6. Tree-Visual Alignment
    print("Calculating Tree-Visual Alignment...")
    align_score = calculate_alignment(model, testloader, device, idx_to_class, root, classes)
    
    cnt = mistake_stats['count'] if mistake_stats['count'] > 0 else 1
    
    print("\n" + "="*40)
    print("HXE METHODOLOGY RESULTS (TIERED IMAGENET)")
    print("="*40)
    print(f"Top-1 Accuracy:            {100*correct/total:.2f}%")
    print("-" * 40)
    print(f"LCA Depth (Metric 1):      {mistake_stats['lca_depth']/cnt:.4f}")
    print(f"Avg Dist to LCA (Metric 2):{mistake_stats['dist']/cnt:.4f}")
    print(f"Rel. LCA Depth (Metric 3): {rel_depth_all/total:.4f}")
    print("-" * 40)
    print(f"Mistake-Only Rel Depth:    {mistake_stats['rel_depth']/cnt:.4f}")
    print(f"Tree-Visual Alignment:     {align_score:.4f}")
    print("="*40)

if __name__ == '__main__':
    main()