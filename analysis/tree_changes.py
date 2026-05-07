import json
import os
import glob
import pandas as pd
import re
import matplotlib.pyplot as plt

def analyze_tree_structure(file_path):
    """
    Parses a single JSON tree file and calculates structural metrics.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Initialize counters
    metrics = {
        "filename": os.path.basename(file_path),
        "height": 0,
        "internal_nodes": 0,
        "leaf_nodes": 0,
        "total_nodes": 0,
        "leaf_depth_sum": 0,
        "branching_sum": 0,
        "max_branching": 0
    }

    # Stack for DFS: (node, depth)
    stack = [(data, 0)]
    
    # Attempt to extract iteration number from filename (e.g., 'tree_1.json' -> 1)
    match = re.search(r'(\d+)', metrics["filename"])
    metrics["iteration"] = int(match.group(1)) if match else 0

    while stack:
        node, depth = stack.pop()
        metrics["total_nodes"] += 1
        
        if depth > metrics["height"]:
            metrics["height"] = depth

        children = node.get("children")
        
        # Check if Leaf or Internal Node
        if not children:
            metrics["leaf_nodes"] += 1
            metrics["leaf_depth_sum"] += depth
        else:
            metrics["internal_nodes"] += 1
            num_children = len(children)
            metrics["branching_sum"] += num_children
            if num_children > metrics["max_branching"]:
                metrics["max_branching"] = num_children
            
            # Add children to stack
            for child in children:
                stack.append((child, depth + 1))

    # Derived Metrics
    if metrics["leaf_nodes"] > 0:
        metrics["avg_leaf_depth"] = metrics["leaf_depth_sum"] / metrics["leaf_nodes"]
    else:
        metrics["avg_leaf_depth"] = 0

    if metrics["internal_nodes"] > 0:
        metrics["avg_branching_factor"] = metrics["branching_sum"] / metrics["internal_nodes"]
    else:
        metrics["avg_branching_factor"] = 0
        
    return metrics

def analyze_folder_and_save(folder_path, output_csv="tree_metrics.csv", output_png="tree_evolution.png"):
    """
    Analyzes all .json files in the folder, saves to CSV, and saves plots to PNG.
    """
    files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not files:
        print("No JSON files found in the specified directory.")
        return

    results = []
    print(f"Found {len(files)} trees. Analyzing...")

    for file in files:
        try:
            metrics = analyze_tree_structure(file)
            results.append(metrics)
        except Exception as e:
            print(f"Error parsing {file}: {e}")

    df = pd.DataFrame(results)
    
    # Sort by iteration
    if "iteration" in df.columns:
        df = df.sort_values("iteration")
    else:
        df = df.sort_values("filename")

    # --- SAVE CSV ---
    df.to_csv(output_csv, index=False)
    print(f"DataFrame saved to {output_csv}")

    # --- SAVE PLOTS ---
    if "iteration" in df.columns and df["iteration"].nunique() > 1:
        x_axis = df["iteration"]
        x_label = "Iteration"
    else:
        x_axis = range(len(df))
        x_label = "File Index"

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Tree Structure Evolution', fontsize=16)

    # Plot 1: Internal Nodes
    axs[0, 0].plot(x_axis, df['internal_nodes'], marker='o', color='b')
    axs[0, 0].set_title('Internal Nodes (Complexity)')
    axs[0, 0].set_xlabel(x_label)
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].grid(True)

    # Plot 2: Avg Leaf Depth
    axs[0, 1].plot(x_axis, df['avg_leaf_depth'], marker='o', color='g')
    axs[0, 1].set_title('Average Leaf Depth (Granularity)')
    axs[0, 1].set_xlabel(x_label)
    axs[0, 1].set_ylabel('Depth')
    axs[0, 1].grid(True)

    # Plot 3: Tree Height
    axs[1, 0].plot(x_axis, df['height'], marker='o', color='r')
    axs[1, 0].set_title('Max Tree Height')
    axs[1, 0].set_xlabel(x_label)
    axs[1, 0].set_ylabel('Height')
    axs[1, 0].grid(True)

    # Plot 4: Avg Branching Factor
    axs[1, 1].plot(x_axis, df['avg_branching_factor'], marker='o', color='purple')
    axs[1, 1].set_title('Avg Branching Factor (Breadth)')
    axs[1, 1].set_xlabel(x_label)
    axs[1, 1].set_ylabel('Avg Children')
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(output_png)
    print(f"Plots saved to {output_png}")

# --- Usage ---
folder_path = './data/logs/final_run_cifar100/trees' 
analyze_folder_and_save(folder_path)