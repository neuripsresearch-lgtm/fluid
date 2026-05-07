import json
import os
from openai import OpenAI, APIStatusError
import time
from typing import List, Optional, Set
from pydantic import BaseModel

# --- Reusing your existing validation logic ---
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
    try:
        tree = TreeNode(**tree_data)
        generated_leaves = tree.get_leaf_nodes()
        if generated_leaves == expected_leaves:
            return True
        else:
            print(f"Validation Error: Missing {expected_leaves - generated_leaves}, Extra {generated_leaves - expected_leaves}")
            return False
    except Exception as e:
        print(f"Pydantic validation error: {e}")
        return False

# --- API Client Setup [cite: 10] ---
client = None
if "GEMINI_API_KEY" in os.environ:
    client = OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

def get_llm_response(prompt, max_tokens=32000, temperature=0.2, model="gemini-2.5-pro"):
    # Using 'thinking' model if available as this single step is complex
    if not client: return None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return None

# --- NEW FUNCTION FOR ABLATION STUDY ---
def refine_tree_directly(current_tree, performance_metrics, misclassifications, prompt_file_path, classes_file_path, model="gemini-2.5-pro", max_retries=3):
    """
    Performs simultaneous critique and editing in a single LLM call.
    """
    print("Refining tree directly (Ablation: No Critic)...")
    
    with open(prompt_file_path, 'r') as f:
        prompt_template = f.read()

    with open(classes_file_path, 'r') as f:
        classes_content = f.read()
    # Expect leaves to match strictly [cite: 46]
    expected_leaves = {cls.strip().replace(' ', '_') for cls in classes_content.strip().split(',')}

    current_tree_str = json.dumps(current_tree, indent=2)
    metrics_str = json.dumps(performance_metrics, indent=2)

    # Construct the massive prompt combining context + data + task
    full_prompt = (
        f"{prompt_template}\n\n"
        f"--- CURRENT STATE ---\n"
        f"Current Hierarchy JSON:\n```json\n{current_tree_str}\n```\n\n"
        f"Performance Metrics:\n```json\n{metrics_str}\n```\n\n"
        f"Significant Misclassifications:\n{misclassifications}\n\n"
        f"--- OUTPUT ---\n"
        "Please provide the corrected JSON object below:"
    )

    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1}/{max_retries} to refine tree...")
        # Higher tokens needed as it outputs the full tree
        response_text = get_llm_response(full_prompt, max_tokens=20000, model=model) 

        if response_text:
            try:
                # Extract JSON from markdown blocks if present [cite: 17]
                if "```json" in response_text:
                    response_text = response_text.split("```json\n")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1]
                
                new_tree = json.loads(response_text)
                
                if validate_tree_leaves(new_tree, expected_leaves):
                    print("Successfully generated valid refined tree.")
                    return new_tree
                else:
                    print("Tree validation failed (leaf mismatch). Retrying...")
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
    
    print("Failed to generate valid tree after retries.")
    return None