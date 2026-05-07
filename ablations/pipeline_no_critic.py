import subprocess
import json
import os
import re
import csv
import shutil
# We import the single-step refiner instead of the critic/editor pair
from llm_handler_ablation import refine_tree_directly

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
    """Parses the output of the evaluation script to extract ALL key metrics."""
    metrics = {}
    misclassifications = []

    try:
        # 1. Top-1 Accuracy
        accuracy_match = re.search(r"Top-1 Accuracy:\s+([\d.]+)%", output_text)
        if accuracy_match:
            metrics['accuracy'] = float(accuracy_match.group(1))

        # 2. Avg LCA Depth (Mistake)
        lca_depth_match = re.search(r"Avg LCA Depth \(Mistake\):\s+([\d.]+)", output_text)
        if lca_depth_match:
            metrics['lca_depth_mistake'] = float(lca_depth_match.group(1))
            
        # 3. Avg Dist to LCA
        avg_dist_match = re.search(r"Avg Dist to LCA:\s+([\d.]+)", output_text)
        if avg_dist_match:
            metrics['avg_dist_to_lca'] = float(avg_dist_match.group(1))
            
        # 4. Rel. LCA Depth (All)
        rel_lca_depth_match = re.search(r"Rel. LCA Depth \(All\):\s+([\d.]+)", output_text)
        if rel_lca_depth_match:
            metrics['relative_lca_depth_all'] = float(rel_lca_depth_match.group(1))

        # --- Hierarchy Distance Metrics ---
        # 5. Hierarchical Dist (Mistake)
        hier_dist_match = re.search(r"Hierarchical Dist \(Mistake\):\s+([\d.]+)", output_text)
        if hier_dist_match:
            metrics['hierarchical_dist_mistake'] = float(hier_dist_match.group(1))

        # 6. Avg Hierarchical Dist @ K=1
        hier_k1_match = re.search(r"Avg Hierarchical Dist @ K=1:\s+([\d.]+)", output_text)
        if hier_k1_match:
            metrics['hierarchical_dist_k1'] = float(hier_k1_match.group(1))

        # 7. Avg Hierarchical Dist @ K=5
        hier_k5_match = re.search(r"Avg Hierarchical Dist @ K=5:\s+([\d.]+)", output_text)
        if hier_k5_match:
            metrics['hierarchical_dist_k5'] = float(hier_k5_match.group(1))

        # 8. Avg Hierarchical Dist @ K=20
        hier_k20_match = re.search(r"Avg Hierarchical Dist @ K=20:\s+([\d.]+)", output_text)
        if hier_k20_match:
            metrics['hierarchical_dist_k20'] = float(hier_k20_match.group(1))

        # 9. Mistake-Only Rel Depth
        mistake_rel_match = re.search(r"Mistake-Only Rel Depth:\s+([\d.]+)", output_text)
        if mistake_rel_match:
            metrics['mistake_only_rel_depth'] = float(mistake_rel_match.group(1))

        # 10. Tree-Visual Alignment
        alignment_match = re.search(r"Tree-Visual Alignment:\s+([-\d.]+)", output_text)
        if alignment_match:
            metrics['tree_visual_alignment'] = float(alignment_match.group(1))

        # Extract significant misclassifications
        misclass_section_match = re.search(r"--- Significant Misclassifications \(Threshold > \d+\) ---\n(.*?)\n---", output_text, re.DOTALL)
        if misclass_section_match:
            misclass_section = misclass_section_match.group(1)
            pattern = r"'([^']*)' → '([^']*)': (\d+) times\s*(?:\n\s*• LCA Depth: ([\d.]+)\s*\n\s*• Avg Dist to LCA: ([\d.]+))?"
            
            for match in re.finditer(pattern, misclass_section):
                true_class, pred_class, count, lca_depth, avg_dist = match.groups()
                misclass_info = {
                    "true": true_class,
                    "predicted": pred_class,
                    "count": int(count)
                }
                if lca_depth and avg_dist:
                    misclass_info['lca_depth'] = float(lca_depth)
                    misclass_info['avg_dist_to_lca'] = float(avg_dist)
                misclassifications.append(misclass_info)
        
        return metrics, misclassifications
    except (IndexError, AttributeError) as e:
        print(f"Error parsing evaluation output: {e}")
        return None, None

def save_metrics_to_csv(metrics_history, output_file):
    if not metrics_history: return
    fieldnames = set()
    for m in metrics_history: fieldnames.update(m.keys())
    fieldnames = sorted(list(fieldnames))
    if 'iteration' in fieldnames:
        fieldnames.remove('iteration')
        fieldnames = ['iteration'] + fieldnames

    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_history)
        print(f"Metrics history saved to {output_file}")
    except IOError as e:
        print(f"Error saving metrics CSV: {e}")

def run_ablation_pipeline(max_iter=10, patience=5):
    """
    Runs the ablation study (Single LLM) with the EXACT same validation/stopping logic
    as the main pipeline.
    """
    # --- Initial Setup ---
    assets_dir = './assets'
    weights_dir = './weights_ablation' # Separate weights dir
    logs_dir = './logs_ablation'       # Separate logs dir
    python_scripts_dir = './python_scripts' 
    
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    classes_file = os.path.join(assets_dir, 'cifar100_classes.txt')
    # The merged prompt for the single LLM
    refine_prompt = os.path.join(assets_dir, 'prompt_single_llm_refine.txt') 
    
    hierarchy_file = os.path.join(assets_dir, 'curr_tree_ablation.json')
    best_hierarchy_file = os.path.join(assets_dir, 'best_tree_ablation.json') # Track best tree
    
    mbm_model_path = os.path.join(weights_dir, 'mbm_model.pth')
    global_best_weights_path = os.path.join(weights_dir, 'mbm_model_global_best.pth')
    
    metrics_csv_path = os.path.join(logs_dir, 'metrics_history.csv')

    # --- Step 1: Generate Initial Tree ---
    if not os.path.exists(hierarchy_file):
        print(f"⚠️ Warning: {hierarchy_file} not found. Please ensure the starting tree matches your main experiment for fair comparison.")
        return 
    else:
        print(f"Using existing tree at {hierarchy_file}")

    # Initialize best tree with current
    shutil.copy(hierarchy_file, best_hierarchy_file)

    # --- Iteration Loop ---
    best_master_metric = 0.0  # Tracks GLOBAL BEST (Validation Set)
    patience_counter = 0
    metrics_history = []

    for i in range(max_iter):
        iteration_num = i + 1
        print(f"\n{'='*80}")
        print(f"ABLATION ITERATION {iteration_num}/{max_iter}")
        
        # --- Configure Training Parameters ---
        # Mimic exact training schedule from pipeline.py
        if iteration_num == 1:
            print("Mode: INITIAL TRAINING (Full Backbone)")
            current_epochs = '50'
            current_lr = '5e-5'
        else:
            print("Mode: REFINEMENT (Frozen Backbone, Last 2 Layers Only)")
            current_epochs = '25'
            current_lr = '5e-5'
            
        print(f"{'='*80}\n")

        # 1. Train MBM Hierarchical Classifier
        # Reuse the exact same training script
        train_cmd = [
            'python', os.path.join(python_scripts_dir, 'train_hierarchical_classifier.py'),
            '--hierarchy-path', hierarchy_file,
            '--save-path', mbm_model_path,
            '--epochs', current_epochs,
            '--lr', current_lr,
            '--patience', '10', 
            '--iteration-num', str(iteration_num),
            '--beta', '1.0'
        ]

        if os.path.exists(global_best_weights_path):
            print(f"Resuming from GLOBAL BEST weights: {global_best_weights_path}")
            train_cmd.extend(['--resume', global_best_weights_path])
        elif os.path.exists(mbm_model_path):
            print(f"Resuming from previous weights: {mbm_model_path}")
            train_cmd.extend(['--resume', mbm_model_path])

        run_command(train_cmd)

        # 2. Evaluate on VALIDATION Split (Crucial for Master Metric)
        print("Evaluating the model on VALIDATION split...")
        eval_output = run_command([
            'python', os.path.join(python_scripts_dir, 'evaluate_hierarchical.py'),
            '--hierarchy-path', hierarchy_file,
            '--model-path', mbm_model_path,
            '--cm-save-path', os.path.join(logs_dir, f'confusion_matrix_val_iter_{iteration_num}.png'),
            '--split', 'val'  # Strict validation split usage
        ])
        
        if not eval_output:
            print("Evaluation failed. Stopping pipeline.")
            break

        metrics, misclassifications = parse_evaluation_output(eval_output)
        
        if not metrics or 'accuracy' not in metrics or 'mistake_only_rel_depth' not in metrics:
            print("Could not parse critical metrics. Stopping.")
            break

        # --- CALCULATE MASTER METRIC (Accuracy * Mistake-Only Rel Depth) ---
        accuracy_val = metrics['accuracy'] / 100.0
        mistake_rel_val = metrics['mistake_only_rel_depth']
        
        current_master_metric = accuracy_val * mistake_rel_val
        metrics['master_metric'] = current_master_metric
        
        print(f"ITERATION {iteration_num} MASTER METRIC: {current_master_metric:.4f}")

        metrics['iteration'] = iteration_num
        metrics['split'] = 'validation'
        metrics_history.append(metrics)
        save_metrics_to_csv(metrics_history, metrics_csv_path)

        # --- Check for improvement vs GLOBAL BEST ---
        if current_master_metric > best_master_metric:
            best_master_metric = current_master_metric
            patience_counter = 0
            print(f"New Global Best Master Metric (Val): {best_master_metric:.4f}. Resetting patience.")
            
            # Save Global Best Weights
            shutil.copy(mbm_model_path, global_best_weights_path)
            
            # Save Global Best Tree
            shutil.copy(hierarchy_file, best_hierarchy_file)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Pipeline patience limit reached. Stopping refinement.")
            break

        # 3. Refine Tree (THE ABLATION STEP)
        # Instead of 'get_editing_instructions' + 'edit_tree', we use the single Direct Refiner.
        with open(hierarchy_file, 'r') as f:
            current_tree = json.load(f)
        
        misclass_str = json.dumps(misclassifications, indent=2)
        
        # Call the ablation handler
        new_tree = refine_tree_directly(
            current_tree=current_tree, 
            performance_metrics=metrics, 
            misclassifications=misclass_str, 
            prompt_file_path=refine_prompt,
            classes_file_path=classes_file
        )

        if not new_tree:
            print("Refinement failed (LLM Error). Skipping update.")
            continue
        
        # Backup and update
        base, ext = os.path.splitext(hierarchy_file)
        old_file_path = f"{base}_{iteration_num}{ext}"
        os.rename(hierarchy_file, old_file_path)
        
        with open(hierarchy_file, 'w') as f:
            json.dump(new_tree, f, indent=2)
        print(f"Tree updated for iteration {iteration_num + 1}")
        
    # --- FINAL TEST EVALUATION ---
    print("\n" + "="*80)
    print("ABLATION FINISHED. RUNNING FINAL TEST EVALUATION")
    print("Using Global Best Weights and Global Best Tree")
    print("="*80 + "\n")
    
    if os.path.exists(global_best_weights_path) and os.path.exists(best_hierarchy_file):
        test_output = run_command([
            'python', os.path.join(python_scripts_dir, 'evaluate_hierarchical.py'),
            '--hierarchy-path', best_hierarchy_file,
            '--model-path', global_best_weights_path,
            '--cm-save-path', os.path.join(logs_dir, 'confusion_matrix_FINAL_TEST.png'),
            '--split', 'test' # Final test on unseen data
        ])
        
        if test_output:
            test_metrics, _ = parse_evaluation_output(test_output)
            if test_metrics:
                acc_t = test_metrics.get('accuracy', 0.0) / 100.0
                mis_t = test_metrics.get('mistake_only_rel_depth', 0.0)
                test_metrics['master_metric'] = acc_t * mis_t
                
                test_metrics['iteration'] = 'FINAL_TEST'
                test_metrics['split'] = 'test'
                metrics_history.append(test_metrics)
                save_metrics_to_csv(metrics_history, metrics_csv_path)
                print(f"FINAL TEST METRICS: {test_metrics}")
    else:
        print("Could not find best model or best tree for final evaluation.")

if __name__ == '__main__':
    run_ablation_pipeline(max_iter=20, patience=5)