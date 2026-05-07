import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ptsemseg.tree import find_depth
from tree_utils import get_node_by_name, get_lca_and_distance, find_path


# ---------------------------------------------------------------------------
# Tree geometry helpers (mirrors the classification evaluate_hierarchical.py)
# ---------------------------------------------------------------------------

def get_node_depth_from_root(root, target, depth=0):
    """DFS to find depth (distance from root) of a given node. O(n)."""
    if root is target:
        return depth
    for child in root.children:
        result = get_node_depth_from_root(child, target, depth + 1)
        if result is not None:
            return result
    return None


def collect_leaf_depths(root, depth=0):
    """Returns a list of depths for every leaf node."""
    if len(root.children) == 0:
        return [depth]
    depths = []
    for child in root.children:
        depths.extend(collect_leaf_depths(child, depth + 1))
    return depths


def get_avg_leaf_depth(root):
    """Average depth of all leaf nodes (mirrors get_average_leaf_depth in classification)."""
    depths = collect_leaf_depths(root)
    return sum(depths) / len(depths) if depths else 1.0


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_segmentation_hierarchy_metrics(confusion_matrix, root, n_classes, class_names,
                                           num_pixels_threshold=1000, iou_scores=None):
    """
    Computes hierarchy-aware metrics from the segmentation confusion matrix,
    pixel-weighted to match the spirit of the per-sample classification metrics.

    Metric alignment:
        lca_depth_mistake       <- avg depth of LCA node (weighted by pixel count)
        avg_dist_to_lca         <- avg (d_true + d_pred - 2*d_lca) / 2  (weighted)
        hierarchical_dist_mistake <- avg lca.height (subtree height below LCA, weighted)
                                   [= find_depth(lca) on ptsemseg nodes]
        mistake_only_rel_depth  <- avg (lca_depth / avg_leaf_depth) for mistakes (weighted)
                                   Higher = mistakes happen nearer to leaves = less severe
    """
    avg_leaf_depth = get_avg_leaf_depth(root)  # normalisation denominator

    total_lca_depth      = 0.0
    total_dist           = 0.0  # sum of d_true + d_pred - 2*d_lca  (= path distance)
    total_lca_height     = 0.0  # sum of find_depth(lca) * pixel_count
    total_rel_depth      = 0.0  # sum of (d_lca / avg_leaf_depth) * pixel_count
    total_mistake_pixels = 0

    misclassifications = []

    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                continue

            pixel_count = int(confusion_matrix[i, j])
            if pixel_count <= 0:
                continue

            class_i_name = class_names[i]
            class_j_name = class_names[j]

            node_i = get_node_by_name(root, class_i_name)
            node_j = get_node_by_name(root, class_j_name)

            if node_i is None or node_j is None:
                continue    # silently skip; node names not in current tree

            lca, path_dist = get_lca_and_distance(root, node_i, node_j)
            if lca is None:
                continue

            # Depth of LCA from root
            lca_path = []
            find_path(root, lca, lca_path)
            d_lca = len(lca_path) - 1           # root is depth 0

            # Height of LCA subtree (equivalent to lca.height in classification)
            lca_height = find_depth(lca)

            # Relative depth of LCA (0 = root, 1 = leaf)
            rel_depth = d_lca / avg_leaf_depth if avg_leaf_depth > 0 else 0.0

            total_lca_depth      += d_lca      * pixel_count
            total_dist           += path_dist  * pixel_count
            total_lca_height     += lca_height * pixel_count
            total_rel_depth      += rel_depth  * pixel_count
            total_mistake_pixels += pixel_count

            if pixel_count >= num_pixels_threshold:
                misclassifications.append({
                    "true":          class_i_name,
                    "predicted":     class_j_name,
                    "count":         pixel_count,
                    "lca_depth":     d_lca,
                    "avg_dist_to_lca": path_dist / 2.0,
                    "lca_height":    lca_height,
                })

    # ------------------------------------------------------------------
    # Aggregate metrics (mirror evaluate_hierarchical.py output names)
    # ------------------------------------------------------------------
    metrics = {}

    if iou_scores is not None:
        metrics['mean_iou'] = float(np.nanmean(list(iou_scores.values())))
    else:
        diag = np.diag(confusion_matrix).sum()
        total = max(1, confusion_matrix.sum())
        metrics['mean_iou'] = float(diag / total)

    if total_mistake_pixels > 0:
        metrics['lca_depth_mistake']       = total_lca_depth      / total_mistake_pixels
        metrics['avg_dist_to_lca']         = (total_dist / 2.0)   / total_mistake_pixels
        metrics['hierarchical_dist_mistake']= total_lca_height     / total_mistake_pixels
        # mistake_only_rel_depth: higher = mistakes near leaves = better hierarchy
        metrics['mistake_only_rel_depth']  = total_rel_depth       / total_mistake_pixels
    else:
        metrics['lca_depth_mistake']        = 0.0
        metrics['avg_dist_to_lca']          = 0.0
        metrics['hierarchical_dist_mistake']= 0.0
        metrics['mistake_only_rel_depth']   = 0.0

    # Master metric: higher mean_iou AND higher mistake_only_rel_depth = better
    metrics['master_metric'] = metrics['mean_iou'] * metrics['mistake_only_rel_depth']

    # Sort by pixel count, keep top-50
    misclassifications = sorted(misclassifications, key=lambda x: x['count'], reverse=True)[:50]

    return metrics, misclassifications


def get_segmentation_editing_instructions(current_tree, metrics, misclassifications, prompt_file_path=None):
    """
    Generates editing instructions from LLM based on segmentation performance.
    Uses Google Vertex AI Gemini model as the critic.
    
    Requirements:
    - Google Cloud credentials (run: gcloud auth application-default login)
    - google-cloud-aiplatform package (pip install google-cloud-aiplatform)
    - Prompt file at the specified path
    
    Args:
        current_tree: Current hierarchy as JSON object
        metrics: Performance metrics dict
        misclassifications: List of high-impact confusions
        prompt_file_path: Path to critic prompt template
    
    Returns:
        Editing instructions JSON or None if LLM unavailable
    """
    if prompt_file_path is None:
        prompt_file_path = os.path.join(os.path.dirname(__file__), 'assets', 'prompt_critic.txt')
    
    try:
        from llm_handler import get_editing_instructions
    except ImportError:
        print("⚠ LLM handler not available. Dynamic hierarchy editing disabled.")
        print("  To enable, install: pip install google-cloud-aiplatform")
        return None
    
    misclass_str = json.dumps(misclassifications, indent=2)
    return get_editing_instructions(current_tree, metrics, misclass_str, prompt_file_path, model="gemini-2.5-pro")


def edit_tree(current_tree, instructions, prompt_file_path=None, classes_fp=None):
    """
    Edits the tree based on LLM-generated instructions.
    Uses Google Vertex AI Gemini model as the editor.
    
    Requirements:
    - Google Cloud credentials (run: gcloud auth application-default login)
    - google-cloud-aiplatform package (pip install google-cloud-aiplatform)
    - Prompt file at the specified path
    - Classes file listing all leaf nodes
    
    Args:
        current_tree: Current hierarchy as JSON object
        instructions: Editing instructions from get_segmentation_editing_instructions
        prompt_file_path: Path to editor prompt template
        classes_fp: Path to file with comma-separated class names
    
    Returns:
        New hierarchy JSON or None if editing failed
    """
    if prompt_file_path is None:
        prompt_file_path = os.path.join(os.path.dirname(__file__), 'assets', 'prompt_main_edit.txt')
    
    try:
        from llm_handler import edit_tree as llm_edit_tree
    except ImportError:
        print("⚠ LLM handler not available. Tree remains unchanged.")
        return None
    
    return llm_edit_tree(current_tree, instructions, prompt_file_path, classes_fp, model="gemini-2.5-pro")


def generate_initial_tree(dataset_name):
    """
    Generates initial hierarchy using LLM (utility function).
    Not used in the main pipeline—hierarchies are pre-defined.
    
    This function is provided for reference if users want to generate
    hierarchies for new datasets.
    """
    print("ℹ Note: generate_initial_tree is not used in the pipeline.")
    print("  Pre-defined hierarchies are loaded from assets/")
    return None
