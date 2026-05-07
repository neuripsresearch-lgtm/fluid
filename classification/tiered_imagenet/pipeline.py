import subprocess
import json
import os
import re
from llm_handler import generate_initial_tree, get_editing_instructions, edit_tree
from utils import download_nltk_data

def run_command(command):
    """Executes a command and returns its output, while also printing it to the console."""
    print(f"Executing: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1,
            universal_newlines=True
        )
        output = ""
        for line in process.stdout:
            print(line, end='')
            output += line
        
        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, output=output)
            
        return output

    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Return code: {e.returncode}")
        print(f"Output:\n{e.output}")
        return None

def parse_evaluation_output(output_text):
    """
    Parses the output of the evaluation script to extract key metrics.
    """
    metrics = {}
    misclassifications = []

    try:
        if not output_text:
            return None, None

        # Extract accuracy
        accuracy_match = re.search(r"Top-1 Accuracy: ([\d.]+)%", output_text)
        if accuracy_match:
            metrics['accuracy'] = float(accuracy_match.group(1))

        # Extract mistake severity
        lca_depth_match = re.search(r"LCA Depth \(Metric 1\): ([\d.]+)", output_text)
        if lca_depth_match:
            metrics['lca_depth'] = float(lca_depth_match.group(1))
            
        avg_dist_match = re.search(r"Avg Dist to LCA \(Metric 2\): ([\d.]+)", output_text)
        if avg_dist_match:
            metrics['avg_dist_to_lca'] = float(avg_dist_match.group(1))
            
        # Extract the new Relative LCA Depth metric
        relative_lca_depth_match = re.search(r"Relative LCA Depth \(Metric 3\): ([\d.]+)", output_text)
        if relative_lca_depth_match:
            metrics['relative_lca_depth'] = float(relative_lca_depth_match.group(1))

        # Extract significant misclassifications
        misclass_section_match = re.search(r"--- Significant Misclassifications \(Threshold > \d+\) ---\n(.*?)\n---", output_text, re.DOTALL)
        if misclass_section_match:
            misclass_section = misclass_section_match.group(1)
            
            pattern = r"'([^']*)' → '([^']*)': (\d+) times"
            
            for match in re.finditer(pattern, misclass_section):
                true_class, pred_class, count = match.groups()
                misclass_info = {
                    "true": true_class,
                    "predicted": pred_class,
                    "count": int(count)
                }
                misclassifications.append(misclass_info)
        
        return metrics, misclassifications
    except (IndexError, AttributeError) as e:
        print(f"Error parsing evaluation output: {e}")
        return None, None


def run_pipeline(max_iter=10, patience=5):
    """
    Runs the full hierarchical classification and tree refinement pipeline for Tiered ImageNet.
    """
    # --- Initial Setup ---
    dataset_name = "tiered_imagenet"
    assets_dir = './assets'
    weights_dir = './weights'
    logs_dir = './logs'
    python_scripts_dir = './python_scripts' 
    data_root = './data/tiered_imagenet_standard' # ROOT directory containing 'train' and 'test' folders
    
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Initialize NLTK
    download_nltk_data()

    classes_file = os.path.join(assets_dir, 'tiered_imagenet_classes.txt')
    initial_prompt = os.path.join(assets_dir, 'prompt_main_generate.txt')
    critic_prompt = os.path.join(assets_dir, 'prompt_critic.txt')
    edit_prompt = os.path.join(assets_dir, 'prompt_main_edit.txt')
    hierarchy_file = os.path.join(assets_dir, 'tiered_imagenet_hierarchy.json')

    # --- Step 1: Generate Initial Tree ---
    if not os.path.exists(hierarchy_file):
        print(f"Generating initial tree for {dataset_name}...")
        initial_tree = generate_initial_tree(classes_file, initial_prompt)
        if not initial_tree:
            print("Failed to generate initial tree. Exiting.")
            return
        with open(hierarchy_file, 'w') as f:
            json.dump(initial_tree, f, indent=2)
    else:
        print(f"Using existing tree at {hierarchy_file}")

    # --- Step 2: Train Swin Transformer Backbone (if needed) ---
    backbone_path = os.path.join(weights_dir, f'{dataset_name}_swin_tiny_backbone.pth')
    if not os.path.exists(backbone_path):
        print("Training Swin Transformer backbone...")
        run_command(['python', '-u', os.path.join(python_scripts_dir, 'train_swin_backbone.py'), 
                     '--data-path', data_root, 
                     '--save-path', backbone_path])

    # --- Iteration Loop ---
    best_relative_lca_depth = 0.0
    patience_counter = 0

    for i in range(max_iter):
        print(f"\n--- Starting Iteration {i+1}/{max_iter} ---")

        # 1. Extract Features
        # UNCOMMENT IF RUNNING PIPELINE FOR THE FIRST TIME WITH A NEW BACKBONE

        # if i == 0:
        #     print("Extracting features from backbone...")
        #     run_command(['python', os.path.join(python_scripts_dir, 'extract_features.py'),
        #                  '--data-path', data_root,
        #                  '--backbone-path', backbone_path])

        # 2. Train Hierarchical Classifier
        print("Training hierarchical classifiers...")
        run_command(['python', '-u', os.path.join(python_scripts_dir, 'train_hierarchical_classifier.py'),
                     '--hierarchy-path', hierarchy_file,
                     '--data-path', data_root])

        # 3. Evaluate
        print("Evaluating the model...")
        eval_output = run_command([
            'python', '-u', os.path.join(python_scripts_dir, 'evaluate_hierarchical.py'),
            '--hierarchy-path', hierarchy_file,
            '--backbone-path', backbone_path,
            '--weights-dir', weights_dir,
            '--data-path', data_root,
            '--cm-save-path', os.path.join(logs_dir, f'confusion_matrix_iter_{i+1}.png')
        ])
        
        if not eval_output:
            print("Evaluation failed. Stopping pipeline.")
            break

        metrics, misclassifications = parse_evaluation_output(eval_output)
        print(f"Iteration {i+1} Metrics: {metrics}")
        
        if not metrics or 'relative_lca_depth' not in metrics:
            print("Could not parse evaluation metrics. Stopping.")
            break

        if metrics['relative_lca_depth'] > best_relative_lca_depth:
            best_relative_lca_depth = metrics['relative_lca_depth']
            patience_counter = 0
            print(f"New best Relative LCA Depth: {best_relative_lca_depth:.4f}. Resetting patience.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Patience limit reached. Stopping the refinement process.")
            break

        # 4. Get Editing Instructions from Critic
        with open(hierarchy_file, 'r') as f:
            current_tree = json.load(f)
        
        misclass_str = json.dumps(misclassifications, indent=2)
        instructions = get_editing_instructions(current_tree, metrics, misclass_str, critic_prompt)

        should_skip_editing = False
        
        if instructions is None:
            print("⚠️ Critic LLM failed. Skipping edit.")
            should_skip_editing = True
        elif len(instructions) == 0:
            print("ℹ️ Critic LLM returned NO instructions. Skipping edit.")
            should_skip_editing = True
            
        if should_skip_editing:
            continue 

        # Save instructions log
        critic_instructions_path = os.path.join(logs_dir, f'critic_instructions_iter_{i+1}.json')
        with open(critic_instructions_path, 'w') as f:
            json.dump(instructions, f, indent=2)

        # 5. Edit the Tree with Main LLM
        new_tree = edit_tree(current_tree, instructions, edit_prompt, classes_file)

        if not new_tree:
            print("⚠️ Failed to edit the tree. Skipping update.")
            continue
        
        # Backup and Save
        base, ext = os.path.splitext(hierarchy_file)
        old_file_path = f"{base}_{i+1}{ext}"
        os.rename(hierarchy_file, old_file_path)
        print(f"Old tree saved to {old_file_path}")
        
        with open(hierarchy_file, 'w') as f:
            json.dump(new_tree, f, indent=2)
        print(f"New tree for iteration {i+2} saved to {hierarchy_file}")
        
    print("\n--- Pipeline Finished ---")

if __name__ == '__main__':
    run_pipeline(max_iter=20, patience=10)