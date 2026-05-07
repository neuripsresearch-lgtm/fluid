import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class Node:
    """A node in the hierarchy tree."""
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.leaf_nodes = []
        self.classifier = None

    def add_child(self, child_node):
        self.children.append(child_node)

def build_tree(data, parent=None):
    """Recursively builds the hierarchy tree from the JSON data."""
    if isinstance(data, str):
        # For leaf nodes, replace spaces with underscores in the name
        node_name = data.replace(' ', '_')
        print(f"Creating leaf node: {data} -> {node_name}")
        node = Node(node_name, parent)
        return node
    
    # For internal nodes, also replace spaces for consistency
    node_name = data.get('name', 'Unnamed Node').replace(' ', '_')
    print(f"Creating internal node: {node_name}")
    node = Node(node_name, parent)

    if 'children' in data and data['children'] is not None: # Check if children is not None
        for child_data in data['children']:
            if child_data: # Make sure child_data is not None
                child_node = build_tree(child_data, node)
                node.add_child(child_node)
    
    return node


def get_leaf_nodes(node):
    """Recursively gets all leaf nodes under a given node."""
    if not node.children:
        return [node.name]
    
    leaf_nodes = []
    for child in node.children:
        leaf_nodes.extend(get_leaf_nodes(child))
    return leaf_nodes

def get_class_to_idx(dataset):
    """Creates a mapping from class names to their indices."""
    print("Creating class to index mapping for the dataset.")
    class_to_idx = {c: i for i, c in enumerate(dataset.classes)}
    print(f"Class to index mapping created with {len(class_to_idx)} classes.")
    return class_to_idx

# gemini/python_scripts/utils.py
import json
import torch
from torch.utils.data import Dataset

# ... (Node class and build_tree function remain the same) ...

class HierarchicalDataset(Dataset):
    """
    A custom dataset for training a classifier at a specific node.
    Works with both original CIFAR-100 (images) and pre-computed features (tensors).
    """
    def __init__(self, original_dataset, class_to_idx, node_leaves):
        self.original_dataset = original_dataset
        self.class_to_idx = class_to_idx
        self.node_leaves = node_leaves
        
        self.indices = []
        self.labels = []
        
        print(f"Creating subset for node leaves: {node_leaves}")

        # Identify which indices belong to the current node
        leaf_indices = set()
        for child_leaves in node_leaves:
            for leaf in child_leaves:
                if leaf in class_to_idx:
                    leaf_indices.add(class_to_idx[leaf])
        
        # --- FAST INITIALIZATION FIX ---
        # Handle both standard Datasets (with .targets) and TensorDatasets (with .tensors)
        if hasattr(original_dataset, 'targets'):
            all_targets = original_dataset.targets
        elif hasattr(original_dataset, 'tensors'):
            # TensorDataset stores labels in the second tensor (index 1)
            all_targets = original_dataset.tensors[1].tolist()
        else:
            # Fallback for other dataset types (slow)
            print("Warning: Slow dataset initialization (no .targets or .tensors found)")
            all_targets = [label for _, label in original_dataset]

        # Filter indices fast
        for i, label in enumerate(all_targets):
            if label in leaf_indices:
                self.indices.append(i)
                self.labels.append(self._get_hierarchical_label(label))
        
        print(f"Subset created with {len(self.indices)} samples.")

    def _get_hierarchical_label(self, original_label):
        # Find original class name from index
        # Note: Inverting dictionary every time is slow, but acceptable for 100 classes.
        # Optimized approach: Pre-compute index-to-child-node mapping if needed.
        original_class = list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(original_label)]
        
        for i, child_leaves in enumerate(self.node_leaves):
            if original_class in child_leaves:
                return i
        return -1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        # This returns (feature_vector, original_label) if using pre-computed features
        # Or (image, original_label) if using raw images
        data, _ = self.original_dataset[original_idx] 
        label = self.labels[idx]
        return data, label