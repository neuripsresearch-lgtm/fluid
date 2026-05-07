import json
import graphviz
import os

os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

def render_tree(json_data, filename):
    """
    Parses a JSON tree and renders it using Graphviz.
    """
    # Create a directed graph
    dot = graphviz.Digraph(comment=filename)
    
    # Set graph attributes for better readability
    dot.attr(rankdir='LR')  # Left to Right layout
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue2', fontname='Helvetica')
    dot.attr('edge', arrowsize='0.5')

    # Recursive function to add nodes and edges
    def add_nodes_edges(node, parent_id=None):
        # specific_id ensures uniqueness if names duplicate across branches
        # We use the memory address id() to ensure every node object is unique in the graph
        node_id = str(id(node))
        
        # specific label from the JSON "name"
        label = node.get('name', 'Unknown')
        
        # Add the node to the graph
        # You can also add the 'justification' as a tooltip if your viewer supports it
        tooltip = node.get('justification', '')
        dot.node(node_id, label, tooltip=tooltip)
        
        # If there is a parent, draw an edge
        if parent_id:
            dot.edge(parent_id, node_id)
        
        # Recurse for children
        children = node.get('children')
        if children:
            for child in children:
                add_nodes_edges(child, node_id)

    # Start the recursion from the root
    add_nodes_edges(json_data)

    # Render the graph to a file (PNG format)
    output_path = dot.render(filename, format='png', cleanup=True)
    print(f"Tree visualized and saved to: {output_path}")

# --- LOAD AND RUN ---

try:
    with open('tree2.json', 'r') as f:
        golden_data = json.load(f)
    render_tree(golden_data, 'visual_tree2')
except FileNotFoundError:
    print("Could not find golden_tree_cifar100.json")