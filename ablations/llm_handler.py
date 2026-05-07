import time
import json
import os
import sys
from pydantic import BaseModel
from typing import List, Optional, Set

# Pydantic Models for Tree Validation
class TreeNode(BaseModel):
    name: str
    justification: str = ""
    children: Optional[List['TreeNode']] = None

    def get_leaf_nodes(self) -> Set[str]:
        if not self.children:
            return {self.name}
        
        leaf_nodes = set()
        for child in self.children:
            leaf_nodes.update(child.get_leaf_nodes())
        return leaf_nodes

TreeNode.update_forward_refs()

def validate_tree_leaves(tree_data: dict, expected_leaves: Set[str]) -> bool:
    """Validates the leaf nodes of the tree against the expected class names."""
    try:
        tree = TreeNode(**tree_data)
        generated_leaves = tree.get_leaf_nodes()
        
        if generated_leaves == expected_leaves:
            print("✓ Leaf node validation successful.")
            return True
        else:
            missing_leaves = expected_leaves - generated_leaves
            extra_leaves = generated_leaves - expected_leaves
            if missing_leaves: print(f"✗ Validation Error: Missing leaf nodes: {missing_leaves}")
            if extra_leaves: print(f"✗ Validation Error: Extra leaf nodes: {extra_leaves}")
            return False
    except Exception as e:
        print(f"✗ Pydantic validation error: {e}")
        return False

def _preprocess_tree_data(node_data):
    """Recursively converts string children to TreeNode-compatible dicts."""
    if isinstance(node_data, dict) and 'children' in node_data and node_data['children']:
        new_children = []
        for child in node_data['children']:
            if isinstance(child, str):
                new_children.append({'name': child, 'justification': '', 'children': None})
            else:
                new_children.append(_preprocess_tree_data(child))
        node_data['children'] = new_children
    return node_data

# --- Vertex AI API Initialization ---
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, SafetySetting, GenerationConfig
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, GoogleAPIError
    from google.auth.exceptions import DefaultCredentialsError

    # Note: Update these with your GCP project details
    GCP_PROJECT = os.environ.get('GCP_PROJECT', 'your-gcp-project-id')
    GCP_LOCATION = os.environ.get('GCP_LOCATION', 'us-central1')
    
    # Initialize Vertex AI
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    print(f"✓ Vertex AI initialized with Project: {GCP_PROJECT}, Location: {GCP_LOCATION}")
    LLM_AVAILABLE = True

except ImportError:
    print("⚠ WARNING: google-cloud-aiplatform not installed. LLM features disabled.")
    print("  To enable LLM integration, install: pip install google-cloud-aiplatform")
    LLM_AVAILABLE = False
except DefaultCredentialsError:
    print("\n" + "!" * 80)
    print("⚠ AUTHENTICATION ERROR: Google Cloud credentials not found.")
    print("To enable LLM features, run: gcloud auth application-default login")
    print("!" * 80 + "\n")
    LLM_AVAILABLE = False
except Exception as e:
    print(f"⚠ Error during Vertex AI initialization: {e}")
    LLM_AVAILABLE = False

def get_llm_response(prompt, max_tokens=32000, temperature=0.2, model_name="gemini-2.5-pro"):
    """Fetches a response from Vertex AI Gemini model with retry logic."""
    
    if not LLM_AVAILABLE:
        print("✗ LLM is not available. Please configure GCP credentials.")
        return None
    
    max_retries = 5
    base_delay = 5.0
    
    try:
        model = GenerativeModel(model_name)
    except Exception as e:
        print(f"✗ Error initializing Vertex AI model '{model_name}': {e}")
        return None

    generation_config = GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )

            if response and hasattr(response, 'text') and response.text:
                return response.text.strip()
            else:
                print("⚠ Vertex AI returned an empty or blocked response.")
                return None
        
        except ResourceExhausted as e:
            retry_time = base_delay * (2 ** attempt)
            if attempt < max_retries - 1:
                print(f"⚠ Rate limited (429). Retrying in {retry_time:.1f}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_time)
                continue
            else:
                print(f"✗ Rate limited after {max_retries} attempts.")
                return None
                
        except ServiceUnavailable as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"⚠ Server unavailable (503). Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            else:
                print(f"✗ Server unavailable after {max_retries} attempts.")
                return None
            
        except DefaultCredentialsError:
            print("✗ Default credentials not found. Please run: gcloud auth application-default login")
            return None
        except GoogleAPIError as e:
            print(f"✗ Vertex AI API error: {e}")
            return None
        except Exception as e:
            print(f"✗ Unexpected error during Vertex AI call: {e}")
            return None

    return None

def generate_initial_tree(classes_file_path, prompt_file_path, model="gemini-2.5-pro", max_retries=3):
    """Generates the initial hierarchical tree using Vertex AI with validation."""
    print(f"Generating initial tree using {model}...")
    
    with open(prompt_file_path, 'r') as f:
        prompt_template = f.read()

    with open(classes_file_path, 'r') as f:
        classes_content = f.read()
    
    expected_leaves = {cls.strip().replace(' ', '_') for cls in classes_content.split(',')}
    full_prompt = f"{prompt_template}\n\nClasses from '{os.path.basename(classes_file_path)}':\n{classes_content}"

    for attempt in range(max_retries):
        print(f"  Attempt {attempt + 1}/{max_retries}...")
        response_text = get_llm_response(full_prompt, model_name=model)

        if response_text:
            try:
                if "```json" in response_text:
                    response_text = response_text.split("```json\n")[1].split("```")[0]
                elif "```" in response_text and response_text.count("```") >= 2:
                    parts = response_text.split("```")
                    for part in parts:
                        try:
                            json.loads(part.strip())
                            response_text = part.strip()
                            break
                        except: continue

                tree_data = json.loads(response_text)
                processed_tree_data = _preprocess_tree_data(tree_data)
                
                if validate_tree_leaves(processed_tree_data, expected_leaves):
                    print("✓ Successfully generated and validated the initial tree.")
                    return processed_tree_data
                print("  Tree validation failed. Retrying...")

            except json.JSONDecodeError as e:
                print(f"  Error decoding JSON: {e}")
    
    print("✗ Failed to generate valid tree after all attempts.")
    return None

def get_editing_instructions(current_tree, performance_metrics, misclassifications, prompt_file_path, model="gemini-2.5-pro"):
    """Gets editing instructions from Vertex AI based on performance."""
    print(f"Getting editing instructions via Vertex AI ({model})...")
    
    with open(prompt_file_path, 'r') as f:
        prompt_template = f.read()

    current_tree_str = json.dumps(current_tree, indent=2)
    performance_metrics_str = json.dumps(performance_metrics, indent=2)

    full_prompt = (
        f"{prompt_template}\n\n"
        f"Current hierarchy:\n```json\n{current_tree_str}\n```\n\n"
        f"Performance metrics:\n```json\n{performance_metrics_str}\n```\n\n"
        f"Misclassifications:\n{misclassifications}\n\n"
        "Please provide editing instructions in JSON format."
    )

    response_text = get_llm_response(full_prompt, temperature=0.1, model_name=model)
    
    if response_text:
        try:
            if "```json" in response_text:
                response_text = response_text.split("```json\n")[1].split("```")[0]
            instructions = json.loads(response_text)
            print("✓ Successfully parsed editing instructions.")
            return instructions
        except (json.JSONDecodeError, TypeError):
            print(f"✗ Error decoding JSON from Critic response.")
    return None

def edit_tree(current_tree, edit_instructions, prompt_file_path, classes_file_path, model="gemini-2.5-pro", max_retries=3):
    """Edits the hierarchical tree using Vertex AI with validation."""
    print(f"Editing the tree via Vertex AI ({model})...")
    
    with open(prompt_file_path, 'r') as f:
        prompt_template = f.read()

    with open(classes_file_path, 'r') as f:
        classes_content = f.read()
    expected_leaves = {cls.strip().replace(' ', '_') for cls in classes_content.strip().split(',')}

    current_tree_str = json.dumps(current_tree, indent=2)
    edit_instructions_str = json.dumps(edit_instructions, indent=2)

    full_prompt = (
        f"{prompt_template}\n\n"
        f"Current hierarchy:\n```json\n{current_tree_str}\n```\n\n"
        f"Edit instructions:\n```json\n{edit_instructions_str}\n```\n\n"
        "Please provide the final modified JSON object."
    )
    
    for attempt in range(max_retries):
        print(f"  Attempt {attempt + 1}/{max_retries}...")
        response_text = get_llm_response(full_prompt, max_tokens=32000, model_name=model)

        if response_text:
            try:
                if "```json" in response_text:
                    response_text = response_text.split("```json\n")[1].split("```")[0]
                new_tree = json.loads(response_text)
                if validate_tree_leaves(new_tree, expected_leaves):
                    print("✓ Successfully edited and validated the new tree.")
                    return new_tree
                print("  Edited tree validation failed. Retrying...")
            except (json.JSONDecodeError, TypeError):
                print("  Error decoding JSON from Editor response.")

    print("✗ Failed to edit tree after all attempts.")
    return None
