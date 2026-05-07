import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import nltk
from nltk.corpus import wordnet as wn

# --- NLTK Helper Functions ---
def download_nltk_data():
    """Ensures necessary NLTK data is downloaded."""
    try:
        nltk.data.find('corpora/wordnet.zip')
        nltk.data.find('corpora/omw-1.4.zip')
    except LookupError:
        print("Downloading NLTK WordNet data...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')

def wnid_to_name(wnid):
    """
    Converts a WNID (e.g., 'n01440764') to a human-readable name (e.g., 'tench')
    using NLTK's WordNet interface.
    """
    try:
        # ImageNet WNIDs are 'n' + 8 digit offset
        offset = int(wnid[1:]) 
        synset = wn.synset_from_pos_and_offset('n', offset)
        # Get the first lemma name (e.g., 'tench')
        return synset.lemmas()[0].name()
    except Exception as e:
        # Fallback if NLTK fails or ID is weird
        # print(f"Warning: Could not map WNID {wnid} to name. Error: {e}")
        return wnid

# --- Tree Node Class ---
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
        node = Node(node_name, parent)
        return node
    
    # For internal nodes, also replace spaces for consistency
    node_name = data.get('name', 'Unnamed Node').replace(' ', '_')
    node = Node(node_name, parent)

    if 'children' in data and data['children'] is not None:
        for child_data in data['children']:
            if child_data:
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

class HierarchicalDataset(Dataset):
    """
    A custom dataset for training a classifier at a specific node.
    """
    def __init__(self, original_dataset, class_to_idx, node_leaves):
        """
        Args:
            original_dataset: The dataset containing features/images.
            class_to_idx: A dictionary mapping the *Human Name* (used in tree) 
                          to the *Index* (used in the tensor dataset).
            node_leaves: A list of lists, where each sublist contains the leaf names 
                         for a child of the current node.
        """
        self.original_dataset = original_dataset
        self.class_to_idx = class_to_idx
        self.node_leaves = node_leaves
        
        self.indices = []
        self.labels = []
        
        # print(f"Creating subset for node leaves: {node_leaves}")

        # Identify which INDICES in the dataset belong to this node.
        # We look up 'human_name' in class_to_idx to get the integer label.
        leaf_indices = set()
        for child_leaves in node_leaves:
            for leaf in child_leaves:
                if leaf in class_to_idx:
                    leaf_indices.add(class_to_idx[leaf])
        
        # Handle both standard Datasets (with .targets) and TensorDatasets (with .tensors)
        if hasattr(original_dataset, 'targets'):
            all_targets = original_dataset.targets
        elif hasattr(original_dataset, 'tensors'):
            # TensorDataset stores labels in the second tensor (index 1)
            all_targets = original_dataset.tensors[1].tolist()
        else:
            # Fallback (slow)
            all_targets = [label for _, label in original_dataset]

        # Filter indices
        for i, label in enumerate(all_targets):
            if label in leaf_indices:
                self.indices.append(i)
                self.labels.append(self._get_hierarchical_label(label))
        
        # print(f"Subset created with {len(self.indices)} samples.")

    def _get_hierarchical_label(self, original_label_idx):
        """
        Maps the original dataset label (int) -> Human Name (str) -> Local Child Index (int).
        """
        # 1. Invert mapping to get Human Name from Index
        # (Optimized: we could pass an inverted dict, but this is fine for ~600 classes)
        original_class_name = None
        for name, idx in self.class_to_idx.items():
            if idx == original_label_idx:
                original_class_name = name
                break
        
        if original_class_name is None:
            return -1

        # 2. Find which child of the current node this Human Name belongs to
        for i, child_leaves in enumerate(self.node_leaves):
            if original_class_name in child_leaves:
                return i
        return -1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        data, _ = self.original_dataset[original_idx] 
        label = self.labels[idx]
        return data, label