import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from utils import build_tree, get_leaf_nodes, HierarchicalDataset, wnid_to_name, download_nltk_data
import argparse
import os

def get_unique_node_name(node):
    """Generates a unique name for the node to avoid collisions (e.g., 'vessel')."""
    if node.parent:
        # Appends parent name to distinguish duplicates like 'vessel' (craft) vs 'vessel' (container)
        return f"{node.name}_via_{node.parent.name}"
    return node.name

def train_node_classifier(node, train_loader, device, num_ftrs):
    """Trains a linear classifier on top of pre-computed features."""
    if not node.children:
        return

    unique_name = get_unique_node_name(node)
    print(f"Training node: {unique_name} (Original: {node.name}) | Children: {len(node.children)}")
    
    classifier = nn.Linear(num_ftrs, len(node.children)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    classifier.train()
    # Fast training on features
    for epoch in range(5): 
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            
    # Save classifier with UNIQUE name
    node.classifier = classifier
    save_path = f"./weights/{unique_name}_classifier.pth"
    torch.save(classifier.state_dict(), save_path)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download_nltk_data()
    
    # 1. Load Hierarchy
    with open(args.hierarchy_path, 'r') as f:
        hierarchy_data = json.load(f)
    root = build_tree(hierarchy_data)
    
    # 2. Load Pre-computed Features
    print("Loading pre-computed features...")
    try:
        train_features = torch.load('./data/features/train_features.pt')
        train_labels = torch.load('./data/features/train_labels.pt')
    except FileNotFoundError:
        print("Error: Feature files not found. Run extract_features.py first!")
        return

    # Create TensorDataset
    feature_dataset = TensorDataset(train_features, train_labels)
    
    # 3. Create Class Mapping (WNID -> Human Name)
    print(f"Reading class list from {args.data_path}...")
    temp_dataset = ImageFolder(os.path.join(args.data_path, 'train'))
    
    wnid_to_idx = temp_dataset.class_to_idx
    
    print("Mapping WNIDs to Human Names using NLTK...")
    human_name_to_idx = {}
    for wnid, idx in wnid_to_idx.items():
        human_name = wnid_to_name(wnid).replace(' ', '_')
        human_name_to_idx[human_name] = idx

    print(f"Mapped {len(human_name_to_idx)} classes.")
    num_ftrs = train_features.shape[1] 

    def traverse_and_train(node):
        if not node.children:
            return
            
        child_leaves = [get_leaf_nodes(child) for child in node.children]
        
        # Pass the HUMAN NAME mapping to the dataset
        node_dataset = HierarchicalDataset(feature_dataset, human_name_to_idx, child_leaves)
        
        if len(node_dataset) > 0:
            node_loader = DataLoader(node_dataset, batch_size=64, shuffle=True, num_workers=0)
            train_node_classifier(node, node_loader, device, num_ftrs)
        
        for child in node.children:
            traverse_and_train(child)

    print("Starting hierarchical training...")
    traverse_and_train(root)
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hierarchy-path', required=True)
    parser.add_argument('--data-path', required=True, help="Path containing 'train' folder")
    args = parser.parse_args()
    main(args)