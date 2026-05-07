from openai import OpenAI, APIStatusError
import time
import json
import os
from pydantic import BaseModel
from typing import List, Optional, Set
import sys

# Pydantic Models for Tree Validation
class TreeNode(BaseModel):
    name: str
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
    """
    Validates the leaf nodes of the tree against the expected class names.
    """
    try:
        tree = TreeNode(**tree_data)
        generated_leaves = tree.get_leaf_nodes()
        
        if generated_leaves == expected_leaves:
            print("Leaf node validation successful.")
            return True
        else:
            missing_leaves = expected_leaves - generated_leaves
            extra_leaves = generated_leaves - expected_leaves
            if missing_leaves:
                print(f"Validation Error: Missing leaf nodes: {missing_leaves}")
            if extra_leaves:
                print(f"Validation Error: Extra leaf nodes: {extra_leaves}")
            return False
            
    except Exception as e:
        print(f"Pydantic validation error: {e}")
        return False


def _preprocess_tree_data(node_data):
    """
    Recursively converts string children to TreeNode-compatible dicts.
    """
    if isinstance(node_data, dict) and 'children' in node_data and node_data['children']:
        new_children = []
        for child in node_data['children']:
            if isinstance(child, str):
                new_children.append({'name': child, 'children': None})
            else:
                new_children.append(_preprocess_tree_data(child))
        node_data['children'] = new_children
    return node_data

# --- API Client Initialization ---
client = None
if "GEMINI_API_KEY" in os.environ:
    client = OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    print("Gemini API client initialized using GEMINI_API_KEY environment variable.")
else:
    print("FATAL ERROR: GEMINI_API_KEY environment variable not set.")
    print("Please export it in your terminal (e.g., export GEMINI_API_KEY='YOUR_KEY') and rerun the script.")
    # Exit or handle gracefully if the client is mandatory for the rest of the script
    # For a complete example, we might raise an error here or let the functions handle `client is None`.


def get_llm_response(prompt, max_tokens=64000, temperature=0.2, model="gemini-3-pro-preview", # Changed default model to gemini
                    reasoning_effort=None, verbosity=None):
    if not client:
        print("OpenAI client not initialized. Cannot proceed.")
        return None

    # --- START: Retry Logic with 429 Retry Delay Extraction ---
    max_retries = 5 
    base_delay = 5.0 # Start with a slightly longer, safer default delay
    
    for attempt in range(max_retries):
        try:
            request_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            # Add parameters specific to the Google-backed API if they were supported
            # Since we are using the OpenAI client wrapper, we stick to standard parameters.
            # Google's model may automatically handle effort/verbosity based on the prompt/model.

            response = client.chat.completions.create(**request_params)

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                finish_reason = response.choices[0].finish_reason if response.choices else "unknown"
                print(f"Warning: The API returned an empty response. Finish reason: '{finish_reason}'")
                return None
        
        except APIStatusError as e:
            
            # --- Handle 429 Resource Exhausted ---
            if e.status_code == 429:
                retry_time = base_delay * (2 ** attempt) # Default exponential backoff
                
                # Try to parse the precise retryDelay from the error body
                try:
                    if e.response:
                        error_body = json.loads(e.response.text)
                        
                        # FIX: Handle case where API returns a list of errors
                        if isinstance(error_body, list):
                            error_body = error_body[0]
                        
                        error_details = error_body.get('error', {}).get('details', [])
                        for detail in error_details:
                            if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                                delay_str = detail.get('retryDelay')
                                if delay_str and delay_str.endswith('s'):
                                    # Convert "10.63s" to float 10.63
                                    requested_delay = float(delay_str[:-1])
                                    # Use the API's requested delay if it's longer than our default backoff
                                    if requested_delay > retry_time:
                                        retry_time = requested_delay
                                    print(f"API requested minimum retry in: {delay_str}. Using: {retry_time:.2f}s")
                                    break
                except Exception as parse_err:
                    # Ignore parsing error and use the exponential backoff `retry_time`
                    print(f"Failed to parse retryDelay from 429 error, using exponential backoff. Error: {parse_err}")

                if attempt < max_retries - 1:
                    print(f"Rate limited (429). Retrying in {retry_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_time + 1.0) # Add 1s buffer to requested delay
                    continue
                else:
                    print(f"Rate limited (429) after {max_retries} attempts. Giving up.")
                    return None
                
            # --- Handle 503 Service Unavailable ---
            elif e.status_code == 503:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Server overloaded (503). Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"Server still overloaded after {max_retries} attempts. Giving up.")
                    return None
            
            # --- Handle Other API Errors ---
            else:
                print(f"An unexpected API error occurred (Status {e.status_code}): {e}")
                return None
                
        except Exception as e:
            # Handle other potential exceptions (network issues, etc.)
            print(f"An unexpected error occurred: {e}")
            return None

    # Should only be reached if retries fail
    return None 

def generate_initial_tree(classes_file_path, prompt_file_path, model="gemini-3-pro-preview", max_retries=3):
    """
    Generates the initial hierarchical tree using the main LLM with validation.
    """
    print("Generating the initial hierarchical tree...")
    with open(prompt_file_path, 'r') as f:
        prompt_template = f.read()

    with open(classes_file_path, 'r') as f:
        classes_content = f.read()
    
    expected_leaves = {cls.strip().replace(' ', '_') for cls in classes_content.split(',')}

    full_prompt = f"{prompt_template}\n\nHere are the classes from '{os.path.basename(classes_file_path)}':\n{classes_content}"

    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1}/{max_retries} to generate a valid tree...")
        response_text = get_llm_response(full_prompt, model=model, reasoning_effort="high", verbosity="medium")

        if response_text:
            try:
                if "```json" in response_text:
                    response_text = response_text.split("```json\n")[1].split("```")[0]
                elif "```" in response_text and response_text.count("```") >= 2:
                    parts = response_text.split("```")
                    for i, part in enumerate(parts):
                        try:
                            json.loads(part.strip())
                            response_text = part.strip()
                            break
                        except json.JSONDecodeError:
                            continue

                tree_data = json.loads(response_text)
                processed_tree_data = _preprocess_tree_data(tree_data)
                
                if validate_tree_leaves(processed_tree_data, expected_leaves):
                    print("Successfully generated and validated the initial tree.")
                    return processed_tree_data
                else:
                    print("Tree validation failed. Retrying...")

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from LLM response: {e}")
                print("LLM Response was:\n", response_text)
    
    print("Failed to generate a valid tree after all retries.")
    return None

def get_editing_instructions(current_tree, performance_metrics, misclassifications, prompt_file_path, model="gemini-3-pro-preview"):
    """
    Gets editing instructions from the critic LLM based on model performance.

    Args:
        current_tree (dict): The current hierarchical tree.
        performance_metrics (dict): A dictionary with keys like 'accuracy', 'lca_depth', etc.
        misclassifications (str): A formatted string of the misclassification data.
        prompt_file_path (str): The path to the prompt_critic.txt file.
        model (str): The model to use (default: "gemini-3-pro-preview")

    Returns:
        dict: The editing instructions as a Python dictionary, or None if an error occurs.
    """
    print("Getting editing instructions from the critic LLM...")
    with open(prompt_file_path, 'r') as f:
        prompt_template = f.read()

    current_tree_str = json.dumps(current_tree, indent=2)
    performance_metrics_str = json.dumps(performance_metrics, indent=2)

    print(f"Performance Metrics: {performance_metrics}")
    print(f"Performance Metrics String: {performance_metrics_str}")

    full_prompt = (
        f"{prompt_template}\n\n"
        f"Here is the current hierarchy:\n```json\n{current_tree_str}\n```\n\n"
        f"Here are the performance metrics:\n```json\n{performance_metrics_str}\n```\n\n"
        f"And here are the most significant misclassifications:\n{misclassifications}\n\n"
        "Please provide the editing instructions in the specified JSON format."
    )

    response_text = get_llm_response(full_prompt, temperature=0.1, model=model,
                                   reasoning_effort="high", verbosity="medium")
    
    print("----------------------------------- Critic LLM Response received ------------------------------------------------")
    print(response_text)
    print("----------------------------------- End of Critic LLM Response ------------------------------------------------")


    if response_text:
        try:
            if "```json" in response_text:
                response_text = response_text.split("```json\n")[1].split("```")[0]
            elif "```" in response_text and response_text.count("```") >= 2:
                parts = response_text.split("```")
                for i, part in enumerate(parts):
                    try:
                        json.loads(part.strip())
                        response_text = part.strip()
                        break
                    except json.JSONDecodeError:
                        continue

            instructions = json.loads(response_text)
            print(f"Successfully generated and parsed editing instructions:\n {instructions}")
            return instructions
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from critic LLM response: {e}")
            print("Critic LLM Response was:\n", response_text)
            return None
    return None

def edit_tree(current_tree, edit_instructions, prompt_file_path, classes_file_path, model="gemini-3-pro-preview", max_retries=3):
    """
    Edits the hierarchical tree based on instructions from the critic LLM with validation.
    """
    print("Editing the tree with the main LLM...")
    with open(prompt_file_path, 'r') as f:
        prompt_template = f.read()

    with open(classes_file_path, 'r') as f:
        classes_content = f.read()
    expected_leaves = {cls.strip().replace(' ', '_') for cls in classes_content.strip().split(',')}

    current_tree_str = json.dumps(current_tree, indent=2)
    edit_instructions_str = json.dumps(edit_instructions, indent=2)

    full_prompt = (
        f"{prompt_template}\n\n"
        f"Here is the current_hierarchy.json:\n```json\n{current_tree_str}\n```\n\n"
        f"And here are the edit_instructions.json:\n```json\n{edit_instructions_str}\n```\n\n"
        "Please provide the final, modified JSON object."
    )
    
    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1}/{max_retries} to edit the tree...")
        response_text = get_llm_response(full_prompt, max_tokens=64000, model=model,
                                   reasoning_effort="medium", verbosity="low")

        if response_text:
            try:
                if "```json" in response_text:
                    response_text = response_text.split("```json\n")[1].split("```")[0]
                elif "```" in response_text and response_text.count("```") >= 2:
                    parts = response_text.split("```")
                    for i, part in enumerate(parts):
                        try:
                            json.loads(part.strip())
                            response_text = part.strip()
                            break
                        except json.JSONDecodeError:
                            continue
                
                new_tree = json.loads(response_text)
                
                if validate_tree_leaves(new_tree, expected_leaves):
                    print("Successfully edited and validated the new tree.")
                    return new_tree
                else:
                    print("Edited tree validation failed. Retrying...")

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from main LLM (edit) response: {e}")
                print("Main LLM Response was:\n", response_text)

    print("Failed to edit the tree and get a valid response after all retries.")
    return None