import json

def load_txt_tree_to_json(filename):
    """
    Converts a tab-indented .txt tree file to a JSON dict.
    The .txt file uses tabs for hierarchy depth; this function wraps everything
    under a synthetic 'root' parent, matching the JSON convention.
    """
    stack = []  # stack of (indent_level, node_dict)
    root = {"name": "root", "justification": "Root of the hierarchy.", "children": []}

    with open(filename, 'r') as f:
        for line in f:
            stripped = line.rstrip('\n')
            if not stripped.strip():
                continue
            indent = len(stripped) - len(stripped.lstrip('\t'))
            name = stripped.strip()
            node = {"name": name, "justification": "", "children": []}

            # Pop stack back to the correct parent level
            while len(stack) > indent:
                stack.pop()

            if len(stack) == 0:
                root["children"].append(node)
            else:
                stack[-1]["children"].append(node)

            stack.append(node)

    return root

def dfs_json_to_txt(node, indent_level, fd):
    name = node.get('name', 'unnamed')
    # If the JSON happens to have "Universal class", skip writing it
    if indent_level == -1:
        # Process children without writing the node itself
        for child in node.get('children', []):
            dfs_json_to_txt(child, indent_level + 1, fd)
        return

    fd.write('\t' * indent_level + name + '\n')
    
    children = node.get('children')
    if children is not None:
        for child in children:
            dfs_json_to_txt(child, indent_level + 1, fd)

def save_json_tree_to_txt(tree_data, filename):
    """
    Converts a JSON tree representation to a tab-indented text file.
    """
    with open(filename, 'w') as f:
        if isinstance(tree_data, dict):
            # If the tree has a dummy root, we don't want to print it
            name = tree_data.get('name', '').lower()
            if name == 'universal class' or name == 'root':
                dfs_json_to_txt(tree_data, -1, f)
            else:
                dfs_json_to_txt(tree_data, 0, f)
        elif isinstance(tree_data, list):
            for child in tree_data:
                dfs_json_to_txt(child, 0, f)

def get_node_by_name(root, name):
    if root.name.lower() == name.lower():
        return root
    for child in root.children:
        res = get_node_by_name(child, name)
        if res:
            return res
    return None

def find_path(root, target, path):
    if root is None:
        return False
    path.append(root)
    if root == target:
        return True
    for child in root.children:
        if find_path(child, target, path):
            return True
    path.pop()
    return False

def get_lca_and_distance(root, node1, node2):
    path1 = []
    path2 = []
    if not find_path(root, node1, path1) or not find_path(root, node2, path2):
        return None, -1
    
    i = 0
    while i < len(path1) and i < len(path2):
        if path1[i] != path2[i]:
            break
        i += 1
    
    lca = path1[i-1]
    dist1 = len(path1) - i
    dist2 = len(path2) - i
    
    # Distance is number of edges between node1 and LCA + node2 and LCA
    return lca, dist1 + dist2
