import argparse
import json
import numpy as np
from sklearn.cluster import SpectralClustering
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def symmetrize_confusion(C: np.ndarray) -> np.ndarray:
    """
    Symmetrize the confusion matrix as per Bengio et al.:
        A = 0.5 * (C + C^T)
    Diagonal entries (correct predictions) are zeroed out so that the
    affinity matrix captures *confusion* only, not class frequency.
    """
    A = 0.5 * (C + C.T)
    np.fill_diagonal(A, 0.0)
    return A


def spectral_partition(indices: list[int], A: np.ndarray, n_clusters: int = 2) -> list[list[int]]:
    """
    Partition a subset of class indices into `n_clusters` groups using
    spectral clustering on the sub-affinity matrix defined by `indices`.

    Falls back to a balanced random split if spectral clustering fails
    (e.g. too few classes) or if all affinities are zero.
    """
    n = len(indices)

    # Base cases: cannot split further
    if n <= n_clusters:
        return [[idx] for idx in indices]

    # Extract sub-matrix for current label set
    sub_A = A[np.ix_(indices, indices)]

    # If affinity matrix is all zeros (no observed confusion), fall back to
    # a balanced random partition — this mirrors the spectral clustering
    # objective which encourages balanced partitions.
    # Use a seeded RNG so the fallback split is deterministic.
    if sub_A.sum() == 0:
        shuffled = indices.copy()
        np.random.default_rng(42).shuffle(shuffled)
        chunk = max(1, n // n_clusters)
        return [shuffled[i * chunk: (i + 1) * chunk] for i in range(n_clusters)]

    try:
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
            n_init=10,
        )
        labels = sc.fit_predict(sub_A)
    except Exception:
        # Hard fallback: balanced split by index order
        chunk = max(1, n // n_clusters)
        labels = np.array([i // chunk for i in range(n)], dtype=int)
        labels = np.clip(labels, 0, n_clusters - 1)

    partitions = []
    for c in range(n_clusters):
        group = [indices[i] for i in range(n) if labels[i] == c]
        if group:  # guard against empty cluster
            partitions.append(group)

    # If clustering collapsed into fewer groups, add any lost indices
    assigned = {idx for group in partitions for idx in group}
    lost = [idx for idx in indices if idx not in assigned]
    if lost:
        partitions[-1].extend(lost)

    return partitions


def auto_node_name(class_names: list[str], depth: int, node_id: int) -> str:
    """
    Generate a human-readable internal node name.
    Used only for intermediate nodes — leaves keep their original class names.
    """
    return f"cluster_d{depth}_n{node_id}"


def auto_justification(class_names: list[str], child_groups: list[list[str]]) -> str:
    """
    Auto-generate a justification string based on which classes are present.
    This is intentionally data-driven (no LLM), mirroring the Bengio et al.
    philosophy of purely confusion-based grouping.
    """
    all_classes = [c for group in child_groups for c in group]
    sample = all_classes[:5]
    suffix = f" and {len(all_classes) - 5} more" if len(all_classes) > 5 else ""
    return (
        f"Spectral partition grouping {len(all_classes)} classes "
        f"({', '.join(sample)}{suffix}) by confusion-matrix affinity."
    )


def build_tree(
    indices: list[int],
    class_names: list[str],
    A: np.ndarray,
    branching_factor: int = 2,
    max_depth: int = 10,
    depth: int = 0,
    node_counter: list = None,
) -> dict | str:
    """
    Recursively build the hierarchy tree (Algorithm 2, Bengio et al.).
    """
    if node_counter is None:
        node_counter = [0]

    # --- Leaf: single class ---
    if len(indices) == 1:
        return class_names[indices[0]]

    # --- Safety: max depth reached, flatten remaining classes ---
    if depth >= max_depth:
        node_counter[0] += 1
        names = [class_names[i] for i in indices]
        return {
            "name": auto_node_name(class_names, depth, node_counter[0]),
            "justification": auto_justification(class_names, [names]),
            "children": names,
        }

    # --- Partition this node's label set ---
    partitions = spectral_partition(indices, A, n_clusters=branching_factor)

    # --- Recurse into each child partition ---
    children = []
    for part in partitions:
        child = build_tree(
            indices=part,
            class_names=class_names,
            A=A,
            branching_factor=branching_factor,
            max_depth=max_depth,
            depth=depth + 1,
            node_counter=node_counter,
        )
        children.append(child)

    node_counter[0] += 1
    child_groups = [
        [c] if isinstance(c, str) else _collect_leaves(c)
        for c in children
    ]

    return {
        "name": auto_node_name(class_names, depth, node_counter[0]),
        "justification": auto_justification(class_names, child_groups),
        "children": children,
    }


def _collect_leaves(node: dict | str) -> list[str]:
    """Recursively collect all leaf class names from a subtree."""
    if isinstance(node, str):
        return [node]
    leaves = []
    for child in node.get("children", []):
        leaves.extend(_collect_leaves(child))
    return leaves


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_tree(root: dict | str, expected_classes: list[str]) -> bool:
    """
    Check that every expected class appears exactly once as a leaf.
    Mirrors the 'Leaf Integrity' constraint from the FH editor prompt.
    """
    leaves = _collect_leaves(root)
    leaf_set = set(leaves)
    expected_set = set(expected_classes)

    missing = expected_set - leaf_set
    extra = leaf_set - expected_set
    duplicates = [c for c in leaves if leaves.count(c) > 1]

    ok = True
    if missing:
        print(f"  [WARN] Missing classes: {missing}")
        ok = False
    if extra:
        print(f"  [WARN] Unexpected classes: {extra}")
        ok = False
    if duplicates:
        print(f"  [WARN] Duplicate leaves: {set(duplicates)}")
        ok = False
    if ok:
        print(f"  [OK] All {len(expected_classes)} classes present exactly once.")
    return ok


def tree_stats(root: dict | str, depth: int = 0) -> dict:
    """Compute basic structural statistics of the generated tree."""
    if isinstance(root, str):
        return {"leaves": 1, "internal": 0, "max_depth": depth, "min_leaf_depth": depth}

    children_stats = [tree_stats(c, depth + 1) for c in root.get("children", [])]
    return {
        "leaves": sum(s["leaves"] for s in children_stats),
        "internal": 1 + sum(s["internal"] for s in children_stats),
        "max_depth": max(s["max_depth"] for s in children_stats),
        "min_leaf_depth": min(s["min_leaf_depth"] for s in children_stats),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_classes(path: str) -> list[str]:
    with open(path) as f:
        content = f.read()
    # Support comma-separated or newline-separated
    if "," in content:
        classes = [c.strip() for c in content.split(",") if c.strip()]
    else:
        classes = [c.strip() for c in content.splitlines() if c.strip()]
    return classes


def make_random_confusion(K: int, seed: int = 0) -> np.ndarray:
    """
    Generate a synthetic confusion matrix for testing.
    Adds structure by giving nearby classes (by index) higher confusion,
    loosely simulating the visual similarity structure of CIFAR-100.
    """
    rng = np.random.default_rng(seed)
    C = rng.integers(0, 5, size=(K, K)).astype(float)
    # Boost confusion within local neighborhoods (simulate visual similarity)
    for i in range(K):
        for j in range(max(0, i - 5), min(K, i + 6)):
            C[i, j] += rng.integers(5, 30)
    np.fill_diagonal(C, 0)
    return C


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a Bengio et al. (2010) label tree from a class list and confusion matrix."
    )
    parser.add_argument(
        "--classes", required=True,
        help="Path to class list file (comma- or newline-separated)."
    )
    parser.add_argument(
        "--confusion_matrix", default=None,
        help="Path to .npy confusion matrix (K x K). Rows=GT, Cols=Predicted."
    )
    parser.add_argument(
        "--random_confusion", action="store_true",
        help="Use a synthetic random confusion matrix (for testing only)."
    )
    parser.add_argument(
        "--output", default="bengio_tree.json",
        help="Output path for the JSON tree (default: bengio_tree.json)."
    )
    parser.add_argument(
        "--branching_factor", type=int, default=2,
        help="Number of children per internal node (default: 2 → binary tree)."
    )
    parser.add_argument(
        "--max_depth", type=int, default=12,
        help="Maximum tree depth safety cap (default: 12)."
    )
    args = parser.parse_args()

    # --- Load classes ---
    classes = load_classes(args.classes)
    K = len(classes)
    print(f"Loaded {K} classes from '{args.classes}'.")

    # --- Load or generate confusion matrix ---
    if args.random_confusion:
        print("Using synthetic random confusion matrix (test mode).")
        C = make_random_confusion(K)
    elif args.confusion_matrix:
        C = np.load(args.confusion_matrix)
        assert C.shape == (K, K), (
            f"Confusion matrix shape {C.shape} does not match K={K}. "
            "Check that the matrix rows/columns follow the same class ordering as the class list."
        )
        print(f"Loaded confusion matrix from '{args.confusion_matrix}'.")
    else:
        raise ValueError(
            "Provide either --confusion_matrix <path.npy> or --random_confusion."
        )

    # --- Symmetrize (Bengio et al. eq.) ---
    A = symmetrize_confusion(C)
    print(f"Symmetrized affinity matrix: shape {A.shape}, "
          f"sum={A.sum():.1f}, non-zero={np.count_nonzero(A)}.")

    # --- Build tree ---
    print(f"\nBuilding tree (branching_factor={args.branching_factor}, "
          f"max_depth={args.max_depth})...")
    root = build_tree(
        indices=list(range(K)),
        class_names=classes,
        A=A,
        branching_factor=args.branching_factor,
        max_depth=args.max_depth,
    )

    # --- Validate ---
    print("\nValidating tree:")
    validate_tree(root, classes)

    # --- Print stats ---
    stats = tree_stats(root)
    print(f"\nTree statistics:")
    print(f"  Leaves        : {stats['leaves']}")
    print(f"  Internal nodes: {stats['internal']}")
    print(f"  Max depth     : {stats['max_depth']}")
    print(f"  Min leaf depth: {stats['min_leaf_depth']}")

    # --- Write JSON ---
    with open(args.output, "w") as f:
        json.dump(root, f, indent=2)
    print(f"\nTree written to '{args.output}'.")


if __name__ == "__main__":
    main()