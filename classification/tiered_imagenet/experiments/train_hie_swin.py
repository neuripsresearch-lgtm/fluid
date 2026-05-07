import torch
import torch.nn as nn
import torch.optim as optim
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
    """Parses the JSON tree with 'children': null fix."""
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

def calculate_alignment(model, loader, device, idx_to_class, tree_root, classes):
    """Calculates Spearman correlation."""
    model.eval()
    class_features = {c: [] for c in classes}
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting features for alignment"):
            images = images.to(device)
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
    
    if len(centroids) < 2: return 0.0
    vis_dists = pdist(np.array(centroids), metric='cosine')
    
    n = len(valid_classes)
    tree_dists = np.zeros((n, n))
    nodes = [find_node(tree_root, c) for c in valid_classes]
    
    for i in range(n):
        for j in range(i+1, n):
            lca = find_lca(nodes[i], nodes[j])
            lca_depth = get_depth(lca) if lca else 0
            d = get_depth(nodes[i]) + get_depth(nodes[j]) - 2 * lca_depth
            tree_dists[i,j] = tree_dists[j,i] = d
            
    rho, _ = spearmanr(vis_dists, squareform(tree_dists))
    return rho

# ==========================================
# 2. DATASET WRAPPERS & VALIDATION
# ==========================================

class CoarseDataset(torch.utils.data.Dataset):
    """
    Wraps a standard dataset but returns (image, parent_label_idx)
    instead of (image, fine_label_idx).
    """
    def __init__(self, dataset, leaf_to_parent_idx):
        self.dataset = dataset
        self.leaf_to_parent_idx = leaf_to_parent_idx
        
    def __getitem__(self, index):
        img, fine_label = self.dataset[index]
        coarse_label = self.leaf_to_parent_idx[fine_label]
        return img, coarse_label

    def __len__(self):
        return len(self.dataset)

def validate(model, loader, device):
    """Validation for the Fine Model (Standard Accuracy)."""
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

def validate_coarse(model, loader, device):
    """Validation for the Coarse Model."""
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

# ==========================================
# 3. MAIN SCRIPT
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hierarchy', default='./assets/golden_tree_tiered_imagenet.json', help='Path to hierarchy.json')
    parser.add_argument('--data-path', default='./data/tiered_imagenet_standard', help='Path to dataset root')
    
    # Fine Model Arguments
    parser.add_argument('--fine-model', default=None, help='Path to pre-trained Flat Swin model (.pth). If NOT provided, one will be trained.')
    parser.add_argument('--fine-epochs', type=int, default=20, help='Epochs to train fine model if training from scratch')
    parser.add_argument('--save-fine-path', default='fine_tiered_best_model.pth', help='Path to save newly trained fine model')
    
    # Coarse Model Arguments
    parser.add_argument('--coarse-epochs', type=int, default=15, help='Epochs to train coarse model')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--save-coarse-path', default='coarse_tiered_best_model.pth')
    
    args = parser.parse_args()
    
    set_seed()
    download_nltk_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Setup Data
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
    
    # Fallback to 'val' if 'test' doesn't exist
    if not os.path.exists(test_dir):
        val_dir = os.path.join(args.data_path, 'val')
        if os.path.exists(val_dir):
            test_dir = val_dir

    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform_test)
    
    # Convert WNIDs to Names
    classes = [wnid_to_name(c).replace(' ', '_') for c in trainset.classes]
    num_classes = len(classes)
    print(f"Detected {num_classes} classes.")
    
    # 2. Build Hierarchy & Map Parents
    print("Parsing Hierarchy...")
    with open(args.hierarchy) as f: 
        tree_data = json.load(f)
    root = build_tree(tree_data)

    leaf_nodes = [find_node(root, c) for c in classes]
    parents = set()
    for n in leaf_nodes:
        if n and n.parent: 
            parents.add(n.parent.name)
        else: 
            parents.add('root')
            
    sorted_parents = sorted(list(parents))
    parent_to_idx = {p: i for i, p in enumerate(sorted_parents)}
    num_coarse_classes = len(sorted_parents)
    
    # Map Fine Class Index (0-N) -> Parent Class Index (0-M)
    leaf_idx_to_parent_idx = {}
    for i, c in enumerate(classes):
        node = find_node(root, c)
        if node and node.parent:
            p_name = node.parent.name
        else:
            p_name = 'root'
        leaf_idx_to_parent_idx[i] = parent_to_idx[p_name]
        
    print(f"Mapped {num_classes} fine classes to {num_coarse_classes} coarse classes.")

    # 3. Train or Load Fine Model
    print("\n--- Phase 1: Fine Model ---")
    fine_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
    fine_model = fine_model.to(device)

    # Standard data loaders for fine training/eval
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.fine_model and os.path.exists(args.fine_model):
        print(f"Loading existing Fine Model from {args.fine_model}...")
        fine_model.load_state_dict(torch.load(args.fine_model, map_location=device))
    else:
        print(f"No valid fine model provided. Training from scratch for {args.fine_epochs} epochs...")
        optimizer_fine = optim.AdamW(fine_model.parameters(), lr=5e-5)
        criterion_fine = nn.CrossEntropyLoss()
        
        best_fine_acc = 0.0
        for epoch in range(args.fine_epochs):
            fine_model.train()
            total_loss = 0
            pbar = tqdm(trainloader, desc=f"Fine Epoch {epoch+1}/{args.fine_epochs}")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer_fine.zero_grad()
                loss = criterion_fine(fine_model(imgs), labels)
                loss.backward()
                optimizer_fine.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            val_acc = validate(fine_model, testloader, device)
            print(f"Epoch {epoch+1} - Fine Acc: {val_acc:.2f}%")
            
            if val_acc > best_fine_acc:
                best_fine_acc = val_acc
                torch.save(fine_model.state_dict(), args.save_fine_path)
        
        print(f"Fine model training complete. Best Acc: {best_fine_acc:.2f}%. Saved to {args.save_fine_path}")
        fine_model.load_state_dict(torch.load(args.save_fine_path)) # Load best for next steps

    fine_model.eval()

    # 4. Train Coarse Model
    print("\n--- Phase 2: Coarse Model ---")
    coarse_trainset = CoarseDataset(trainset, leaf_idx_to_parent_idx)
    coarse_testset = CoarseDataset(testset, leaf_idx_to_parent_idx)
    
    coarse_loader = torch.utils.data.DataLoader(coarse_trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    coarse_val_loader = torch.utils.data.DataLoader(coarse_testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    coarse_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_coarse_classes)
    coarse_model = coarse_model.to(device)
    
    optimizer = optim.AdamW(coarse_model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_coarse_acc = 0.0
    for epoch in range(args.coarse_epochs):
        coarse_model.train()
        total_loss = 0
        pbar = tqdm(coarse_loader, desc=f"Coarse Epoch {epoch+1}/{args.coarse_epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(coarse_model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validation
        val_acc = validate_coarse(coarse_model, coarse_val_loader, device)
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(coarse_loader):.4f} | Coarse Acc: {val_acc:.2f}%")
        
        if val_acc > best_coarse_acc:
            print(f"--> New Best Coarse Model! Saving to {args.save_coarse_path}")
            best_coarse_acc = val_acc
            torch.save(coarse_model.state_dict(), args.save_coarse_path)
            
    # 5. Ensemble Inference
    print("\n--- Phase 3: Ensemble Inference (HiE) ---")
    
    # Load Best Coarse Model
    coarse_model.load_state_dict(torch.load(args.save_coarse_path))
    coarse_model.eval()
    
    correct = 0
    total = 0
    mistake_stats = {'lca_depth': 0, 'rel_depth': 0, 'dist': 0, 'count': 0}
    rel_depth_all = 0
    
    idx_to_class = {i: c for i,c in enumerate(classes)}
    leaves = [find_node(root, c) for c in classes]
    valid_leaves = [l for l in leaves if l is not None]
    avg_leaf_depth = sum(get_depth(l) for l in valid_leaves) / len(valid_leaves) if valid_leaves else 1.0

    with torch.no_grad():
        for imgs, lbls in tqdm(testloader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            # Get probabilities
            fine_logits = fine_model(imgs)
            coarse_logits = coarse_model(imgs)
            
            p_fine = torch.softmax(fine_logits, dim=1)
            p_coarse = torch.softmax(coarse_logits, dim=1)
            
            # COMBINATION STEP:
            # P(leaf) = P_fine(leaf) * P_coarse(parent(leaf))
            
            # Build multiplier matrix for the batch
            multiplier = torch.zeros_like(p_fine)
            for k in range(num_classes):
                parent_idx = leaf_idx_to_parent_idx[k]
                multiplier[:, k] = p_coarse[:, parent_idx]
            
            final_probs = p_fine * multiplier
            preds = final_probs.argmax(dim=1)
            
            # Metrics Calculation
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
                        dist_t = d_t - lca_d
                        dist_p = d_p - lca_d
                        mistake_stats['dist'] += (dist_t + dist_p) / 2.0

    # 6. Tree-Visual Alignment (using Fine Model Features)
    print("Calculating Tree-Visual Alignment...")
    align_score = calculate_alignment(fine_model, testloader, device, idx_to_class, root, classes)
    
    cnt = mistake_stats['count'] if mistake_stats['count'] > 0 else 1
    
    print("\n" + "="*40)
    print("HiE (ENSEMBLE) RESULTS (TIERED IMAGENET)")
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