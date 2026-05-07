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
from tqdm import tqdm

# --- Tree Helper Classes (Adapted from utils.py) ---
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

def get_lca_depth(node1, node2):
    """Returns the depth of the Lowest Common Ancestor (root=0)."""
    path1 = []
    curr = node1
    while curr:
        path1.append(curr)
        curr = curr.parent
    
    path2 = set()
    curr = node2
    while curr:
        path2.add(curr)
        curr = curr.parent
        
    for node in path1:
        if node in path2:
            # Calculate depth of this LCA
            depth = 0
            d_curr = node
            while d_curr.parent:
                depth += 1
                d_curr = d_curr.parent
            return depth
    return 0

def get_tree_distance(node1, node2):
    """Calculates distance between two nodes: dist(a, b) = depth(a) + depth(b) - 2*depth(lca)"""
    # 1. Get depths
    d1 = 0
    c1 = node1
    while c1.parent: d1 += 1; c1 = c1.parent
    
    d2 = 0
    c2 = node2
    while c2.parent: d2 += 1; c2 = c2.parent
    
    # 2. Get LCA depth
    lca_d = get_lca_depth(node1, node2)
    
    return d1 + d2 - 2 * lca_d

# --- Hierarchical Loss Logic ---

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
        root = build_tree(tree_data)
        
        # 2. Map class indices to Tree Nodes
        self.nodes = []
        for class_name in classes:
            clean_name = class_name.replace(' ', '_')
            node = find_node(root, clean_name)
            if not node:
                raise ValueError(f"Class '{clean_name}' not found in hierarchy!")
            self.nodes.append(node)
            
        # 3. Pre-compute Distance Matrix (N x N)
        dist_matrix = np.zeros((self.num_classes, self.num_classes))
        print("Pre-computing tree distance matrix...")
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i == j:
                    dist_matrix[i, j] = 0
                else:
                    dist_matrix[i, j] = get_tree_distance(self.nodes[i], self.nodes[j])
        
        # 4. Convert Distances to Soft Probabilities
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
        # Get the soft targets for this batch
        target_dist = self.soft_labels[targets]
        
        # Standard KL Divergence Loss
        # log_softmax(logits) vs soft_targets
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.kl_div(log_probs, target_dist, reduction='batchmean')
        return loss

# --- Main Training Script ---

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Setup (Standard CIFAR-100)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    testset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. Model Setup
    print("Creating Swin Transformer...")
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=100)
    model = model.to(device)

    # 3. Loss Function Setup
    # Get class list from dataset to ensure correct ordering
    classes = trainset.classes
    criterion = HierarchicalSoftLoss(
        hierarchy_file=args.hierarchy_path,
        classes=classes,
        beta=args.beta,
        device=device
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Training Loop
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Use our custom Hierarchical Loss
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

        # Validation (Standard Accuracy)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")
        
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            print(f"New Best Model! Saving to {args.save_path}")
            torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='./data', type=str)
    parser.add_argument('--hierarchy-path', default='./assets/curr_tree.json', type=str, help="Path to your generated hierarchy")
    parser.add_argument('--save-path', default='cifar100_mbm_swin.pth', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--weight-decay', default=0.05, type=float)
    parser.add_argument('--beta', default=1.0, type=float, help="Softness parameter. Lower = Softer, Higher = Harder (Closer to CrossEntropy)")
    
    args = parser.parse_args()
    main(args)