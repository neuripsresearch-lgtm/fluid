import json
import os
import sys
import numpy as np
from scipy.spatial.distance import cosine
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, pearsonr # <-- Added pearsonr
from tqdm import tqdm

# --- Configuration ---
MY_TREE_FILE = './assets/curr_tree.json'
HAF_TREE_FILE = './assets/tiered_imagenet_hierarchy.json'
CLASSES_FILE = './assets/tiered_imagenet_classes.txt'
GLOVE_FILE = './assets/glove.6B.100d.txt' 
# ---------------------

def load_classes(filename: str) -> list[str]:
    """Loads the class list from the text file."""
    print(f"Loading class list from {filename}...")
    with open(filename, 'r') as f:
        content = f.read()
    # Handle the prefix
    if ']' in content:
        content = content.split(']', 1)[-1]
    classes = [c.strip() for c in content.split(',') if c.strip()]
    print(f"Found {len(classes)} total classes in file.")
    return classes

def load_glove_model(glove_file: str) -> dict[str, np.ndarray]:
    """Loads the GloVe model from a text file into a dictionary."""
    if not os.path.exists(glove_file):
        print(f"Error: GloVe file not found at {glove_file}")
        print("Please download 'glove.6B.zip' from https://nlp.stanford.edu/projects/glove/")
        print("Extract it and place 'glove.6B.100d.txt' in the same directory.")
        sys.exit(1)
        
    print(f"Loading GloVe model from {glove_file}...")
    model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        # Get file size for TQDM progress bar
        file_size = os.path.getsize(glove_file)
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Loading GloVe") as pbar:
            for line in f:
                pbar.update(len(line.encode('utf-8')))
                split_line = line.split()
                word = split_line[0]
                try:
                    embedding = np.array([float(val) for val in split_line[1:]])
                    model[word] = embedding
                except ValueError:
                    print(f"Skipping malformed line for word: {word}")
    print(f"Loaded {len(model)} words into GloVe model.")
    return model

def get_embedding(class_name: str, glove_model: dict) -> np.ndarray | None:
    """
    Gets the embedding for a class.
    Handles multi-word classes (e.g., 'aquarium_fish') by averaging vectors.
    """
    words = class_name.split('_')
    vectors = []
    for word in words:
        vec = glove_model.get(word)
        if vec is not None:
            vectors.append(vec)
    
    if not vectors:
        return None # Word(s) not in GloVe
    
    return np.mean(vectors, axis=0)

def build_tree_maps(tree: dict) -> tuple[dict, dict, dict]:
    """
    Performs a DFS on the tree to build parent and depth maps.
    Handles both dictionary nodes (intermediate) and string nodes (leaves).
    """
    parent_map = {}  # {child_key: parent_key}
    depth_map = {}   # {key: depth}
    name_to_key_map = {} # {leaf_name: key}
    
    unique_id = 0

    def dfs(node, parent_key, depth):
        nonlocal unique_id
        
        # --- FIX: Handle string leaf nodes ---
        if isinstance(node, str):
            name = node
            children = None
        else:
            name = node['name']
            children = node.get('children')
        # -------------------------------------

        # Every node gets a unique key, e.g., "worm_ID_55"
        my_key = f"{name}_ID_{unique_id}"
        unique_id += 1
        
        parent_map[my_key] = parent_key
        depth_map[my_key] = depth
        
        if children:
            for child in children:
                dfs(child, my_key, depth + 1)
        else:
            # This is a leaf node. Map its name to its unique key.
            # If a name appears multiple times, the last one visited overwrites the map.
            name_to_key_map[name] = my_key
            
    dfs(tree, None, 0) # Start traversal
    return parent_map, depth_map, name_to_key_map

def get_path_distance(class1: str, class2: str, parent_map: dict, depth_map: dict, name_to_key_map: dict) -> int:
    """
    Calculates the path distance between two nodes using their
    Lowest Common Ancestor (LCA), based on unique keys.
    Distance = depth(c1) + depth(c2) - 2 * depth(LCA)
    """
    
    # 0. Get the unique keys for the leaf classes
    key1 = name_to_key_map.get(class1)
    key2 = name_to_key_map.get(class2)
    
    if not key1:
        raise Exception(f"Could not find leaf node key for class: {class1}")
    if not key2:
        raise Exception(f"Could not find leaf node key for class: {class2}")

    # 1. Find path from key1 to root
    path1 = set()
    curr_key = key1
    while curr_key:
        path1.add(curr_key)
        curr_key = parent_map.get(curr_key) # Get parent's key

    # 2. Find LCA by tracing key2's path up
    curr_key = key2
    lca_key = None
    while curr_key:
        if curr_key in path1:
            lca_key = curr_key
            break
        curr_key = parent_map.get(curr_key) # Get parent's key

    if lca_key is None:
        # This should not happen if both classes are in the tree
        raise Exception(f"Could not find LCA for {class1} and {class2}")

    # 3. Calculate distance
    dist = depth_map[key1] + depth_map[key2] - 2 * depth_map[lca_key]
    return dist

def create_embedding_distance_matrix(classes: list[str], glove_model: dict) -> np.ndarray:
    """Creates the N x N cosine distance matrix from GloVe embeddings."""
    n = len(classes)
    dist_matrix = np.zeros((n, n))
    
    # Get all vectors first
    vector_list = [get_embedding(c, glove_model) for c in classes]
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = cosine(vector_list[i], vector_list[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    return dist_matrix

def create_tree_distance_matrix(tree: dict, classes: list[str]) -> np.ndarray:
    """Creates the N x N path distance matrix from a tree."""
    n = len(classes)
    dist_matrix = np.zeros((n, n))
    
    # Build maps for efficient distance calculation
    parent_map, depth_map, name_to_key_map = build_tree_maps(tree)
    
    # Verify all classes are in the name_to_key_map
    for c in classes:
        if c not in name_to_key_map:
            raise Exception(f"Class '{c}' from class list was not found as a leaf node in the tree.")
    
    for i in range(n):
        for j in range(i + 1, n):
            class1 = classes[i]
            class2 = classes[j]
            # Use the new get_path_distance function
            dist = get_path_distance(class1, class2, parent_map, depth_map, name_to_key_map)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    return dist_matrix

def print_results(corr_my_spearman, corr_haf_spearman, corr_my_pearson, corr_haf_pearson):
    """Prints the final comparison results for both Spearman and Pearson."""
    print("\n--- 📊 Final Results ---")
    
    # --- Spearman ---
    print("\nSpearman's Rank Correlation (ρ) - (Measures monotonic relationship)")
    print("-" * 70)
    score_my_s = corr_my_spearman.correlation
    score_haf_s = corr_haf_spearman.correlation
    
    print(f"Your Tree (curr_tree.json): ρ = {score_my_s:.6f} (p-value: {corr_my_spearman.pvalue:.2e})")
    print(f"OG Tree (tiered_imagenet_hierarchy.json):   ρ = {score_haf_s:.6f} (p-value: {corr_haf_spearman.pvalue:.2e})")
    
    if score_my_s > score_haf_s:
        diff_s = ((score_my_s - score_haf_s) / abs(score_haf_s)) * 100
        print(f"🏆 **Conclusion (Spearman): Your tree is {diff_s:.2f}% more semantically coherent.**")
    elif score_haf_s > score_my_s:
        diff_s = ((score_haf_s - score_my_s) / abs(score_my_s)) * 100
        print(f"Conclusion (Spearman): The HAF tree is {diff_s:.2f}% more semantically coherent.")
    else:
        print("Conclusion (Spearman): Both trees have identical rank-order coherence scores.")

    # --- Pearson ---
    print("\nPearson's Linear Correlation (r) - (Measures linear relationship)")
    print("-" * 70)
    score_my_p = corr_my_pearson.correlation
    score_haf_p = corr_haf_pearson.correlation

    print(f"Your Tree (curr_tree.json): r = {score_my_p:.6f} (p-value: {corr_my_pearson.pvalue:.2e})")
    print(f"HAF Tree (tree_haf.json):   r = {score_haf_p:.6f} (p-value: {corr_haf_pearson.pvalue:.2e})")

    if score_my_p > score_haf_p:
        diff_p = ((score_my_p - score_haf_p) / abs(score_haf_p)) * 100
        print(f"🏆 **Conclusion (Pearson): Your tree's structure has a {diff_p:.2f}% stronger linear correlation.**")
    elif score_haf_p > score_my_p:
        diff_p = ((score_haf_p - score_my_p) / abs(score_my_p)) * 100
        print(f"Conclusion (Pearson): The HAF tree's structure has a {diff_p:.2f}% stronger linear correlation.")
    else:
        print("Conclusion (Pearson): Both trees have identical linear correlation scores.")
        
    print("\n" + "="*70)
    print("Overall Winner (based on Spearman's ρ):")
    if score_my_s > score_haf_s:
        print("Your tree (curr_tree.json) is the most semantically coherent.")
    else:
         print("The HAF tree (tree_haf.json) is the most semantically coherent.")
    print("="*70)


def main():
    # 1. Load all data
    base_class_list = load_classes(CLASSES_FILE)
    glove_model = load_glove_model(GLOVE_FILE)
    
    with open(MY_TREE_FILE, 'r') as f:
        my_tree = json.load(f)
        
    with open(HAF_TREE_FILE, 'r') as f:
        haf_tree = json.load(f)

    # 2. Find the set of classes common to all sources
    print("\nFinding common classes (filtering by GloVe availability)...")
    
    glove_classes = set()
    for c in base_class_list:
        if get_embedding(c, glove_model) is not None:
            glove_classes.add(c)
        else:
            print(f"  [Info] Excluding '{c}': Not found in GloVe model.")

    # we only need to filter by what's in GloVe.
    common_classes = set(base_class_list) & glove_classes
    
    final_class_list = sorted(list(common_classes))
    
    if not final_class_list:
        print("Error: No common classes found between the trees and the GloVe model. Exiting.")
        return
        
    print(f"\nProceeding with {len(final_class_list)} common classes (out of {len(base_class_list)}).")

    # 3. Create the distance matrices
    print("Creating Embedding Distance Matrix (D_Emb)...")
    D_Emb = create_embedding_distance_matrix(final_class_list, glove_model)
    
    print("Creating Your Tree Distance Matrix (D_MyTree)...")
    D_MyTree = create_tree_distance_matrix(my_tree, final_class_list)
    
    print("Creating HAF Tree Distance Matrix (D_HafTree)...")
    D_HafTree = create_tree_distance_matrix(haf_tree, final_class_list)

    # 4. Flatten matrices to vectors (using upper triangle)
    print("Flattening matrices...")
    vec_emb = squareform(D_Emb)
    vec_my_tree = squareform(D_MyTree)
    vec_haf_tree = squareform(D_HafTree)
    
    # 5. Calculate and print correlations
    print("Calculating correlations...")
    
    # Spearman
    corr_my_spearman = spearmanr(vec_emb, vec_my_tree)
    corr_haf_spearman = spearmanr(vec_emb, vec_haf_tree)
    
    # Pearson
    corr_my_pearson = pearsonr(vec_emb, vec_my_tree)
    corr_haf_pearson = pearsonr(vec_emb, vec_haf_tree)
    
    # Print both
    print_results(corr_my_spearman, corr_haf_spearman, corr_my_pearson, corr_haf_pearson)


if __name__ == "__main__":
    main()