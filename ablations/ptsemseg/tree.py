"""
Tree structure and hierarchy utilities for semantic segmentation.
Used for building and manipulating class hierarchies from text/JSON files.
"""

import json
from typing import List, Dict, Optional, Any


class TreeNode:
    """
    Represents a node in a hierarchical tree (for class hierarchy in segmentation).
    Mirrors the Node class from classification/utils.py.
    """
    def __init__(self, name: str, parent: Optional['TreeNode'] = None):
        self.name = name
        self.parent = parent
        self.children: List['TreeNode'] = []
        self.channel = None  # For segmentation: channel index or class index
        self.level = None    # Depth level in tree
        self.leaf_nodes = []  # List of leaf node names under this node
        
    def add_child(self, child: 'TreeNode'):
        """Add a child node to this node's children."""
        self.children.append(child)
        child.parent = self
        
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (terminal node)."""
        return len(self.children) == 0


def find_depth(root: TreeNode) -> int:
    """
    Find the maximum depth of the tree starting from a given root node.
    Depth = max distance from root to any leaf.
    """
    if root.is_leaf():
        return 0
    
    if not root.children:
        return 0
    
    max_child_depth = max(find_depth(child) for child in root.children)
    return max_child_depth + 1


def create_tree_from_textfile(filepath: str) -> TreeNode:
    """
    Create a tree hierarchy from a text file.
    Expected format: hierarchical indentation or specific tree format.
    Example:
        Root
            Child1
                Grandchild1
            Child2
    """
    with open(filepath, 'r') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]
    
    if not lines:
        return TreeNode("root")
    
    root = None
    stack = []  # Stack of (indentation_level, node)
    
    for line in lines:
        if not line.strip():
            continue
            
        # Calculate indentation level
        indent = len(line) - len(line.lstrip())
        indent_level = indent // 4  # Assuming 4-space indentation
        node_name = line.strip()
        
        # Create new node
        new_node = TreeNode(node_name)
        
        if root is None:
            root = new_node
            stack = [(indent_level, root)]
        else:
            # Pop stack until we find the parent level
            while stack and stack[-1][0] >= indent_level:
                stack.pop()
            
            if stack:
                parent_node = stack[-1][1]
                parent_node.add_child(new_node)
            
            stack.append((indent_level, new_node))
    
    return root if root else TreeNode("root")


def add_channels(root: TreeNode, start_idx: int = 0) -> int:
    """
    Assign channel indices (class IDs) to all leaf nodes in the tree.
    Returns the total number of classes (leaves).
    """
    leaf_idx = start_idx
    
    def assign_channels_dfs(node: TreeNode):
        nonlocal leaf_idx
        if node.is_leaf():
            node.channel = leaf_idx
            leaf_idx += 1
        else:
            for child in node.children:
                assign_channels_dfs(child)
    
    assign_channels_dfs(root)
    return leaf_idx


def add_levels(root: TreeNode, max_depth: int):
    """
    Assign depth level (0 to max_depth) to each node in the tree.
    Level represents distance from root.
    """
    def assign_levels_dfs(node: TreeNode, level: int = 0):
        node.level = level
        for child in node.children:
            assign_levels_dfs(child, level + 1)
    
    assign_levels_dfs(root, 0)


def update_channels(root: TreeNode, class_mapping: List[int]):
    """
    Update channel assignments for leaf nodes based on a mapping.
    class_mapping: list where class_mapping[original_leaf_idx] = dataset_class_idx
    
    Example: If tree has leaves [0, 1, 2] but dataset maps them to [0, 10, 7],
    then class_mapping = [0, 10, 7]
    """
    leaves = []
    
    def collect_leaves_dfs(node: TreeNode):
        if node.is_leaf():
            leaves.append(node)
        else:
            for child in node.children:
                collect_leaves_dfs(child)
    
    collect_leaves_dfs(root)
    
    # Update channels
    for i, leaf in enumerate(leaves):
        if i < len(class_mapping):
            leaf.channel = class_mapping[i]


def get_leaf_nodes(node: TreeNode) -> List[str]:
    """Get names of all leaf nodes under this node."""
    if not node.children:
        return [node.name]
    
    leaf_names = []
    for child in node.children:
        leaf_names.extend(get_leaf_nodes(child))
    return leaf_names


def get_all_nodes(node: TreeNode) -> List[TreeNode]:
    """Get all nodes in subtree (DFS)."""
    nodes = [node]
    for child in node.children:
        nodes.extend(get_all_nodes(child))
    return nodes


def build_tree_from_json(data: Dict[str, Any], parent: Optional[TreeNode] = None) -> TreeNode:
    """
    Build a tree from JSON dict representation (matches classification utils).
    """
    if isinstance(data, str):
        node_name = data.replace(' ', '_')
        return TreeNode(node_name, parent)
    
    node_name = data.get('name', 'Unnamed Node').replace(' ', '_')
    node = TreeNode(node_name, parent)
    
    if 'children' in data and data['children'] is not None:
        for child_data in data['children']:
            if child_data:
                child_node = build_tree_from_json(child_data, node)
                node.add_child(child_node)
    
    return node


def get_node_by_name(root: TreeNode, target_name: str) -> Optional[TreeNode]:
    """Search for a node by name in the tree (DFS)."""
    if root.name == target_name:
        return root
    
    for child in root.children:
        result = get_node_by_name(child, target_name)
        if result:
            return result
    
    return None


def get_lca(node_a: TreeNode, node_b: TreeNode) -> Optional[TreeNode]:
    """
    Get the Lowest Common Ancestor (LCA) of two nodes.
    """
    # Get ancestors of node_a
    ancestors_a = set()
    current = node_a
    while current:
        ancestors_a.add(current)
        current = current.parent
    
    # Traverse up from node_b until we find a common ancestor
    current = node_b
    while current:
        if current in ancestors_a:
            return current
        current = current.parent
    
    return None


def get_distance(node_a: TreeNode, node_b: TreeNode) -> int:
    """Get distance (number of edges) between two nodes in the tree."""
    lca = get_lca(node_a, node_b)
    if not lca:
        return -1
    
    dist_a = 0
    current = node_a
    while current != lca:
        dist_a += 1
        current = current.parent
    
    dist_b = 0
    current = node_b
    while current != lca:
        dist_b += 1
        current = current.parent
    
    return dist_a + dist_b
