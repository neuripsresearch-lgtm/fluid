import re
import pandas as pd
import os

def extract_metrics_to_excel(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    # specific regex patterns to find the values
    patterns = {
        "Total Samples": r"Total Samples:\s+(\d+)",
        "Correct Predictions": r"Correct Predictions:\s+(\d+)",
        "Top-1 Accuracy (%)": r"Top-1 Accuracy:\s+([\d\.]+)",
        "LCA Depth (Metric 1)": r"LCA Depth \(Metric 1\):\s+([\d\.]+)",
        "Avg Dist to LCA (Metric 2)": r"Avg Dist to LCA \(Metric 2\):\s+([\d\.]+)",
        "Relative LCA Depth (Metric 3)": r"Relative LCA Depth \(Metric 3\):\s+([\d\.]+)",
        "Mistake-Only Rel Depth": r"Mistake-Only Relative Depth:\s+([\d\.]+)",
        "Tree-Visual Alignment": r"Tree-Visual Alignment \(Spearman\):\s+([-\d\.]+)" # Handles negative numbers
    }

    extracted_data = []

    try:
        with open(input_file, 'r') as f:
            content = f.read()

        # Split the file content by the "Evaluation Finished" marker
        # This creates chunks of text, where each chunk (after the first) represents one evaluation block
        sections = content.split("--- Evaluation Finished ---")

        # Skip the first section as it contains text *before* the first evaluation marker
        for i, section in enumerate(sections[1:], 1):
            row_data = {"Evaluation_ID": i} # Add an ID to track iteration order
            found_any = False

            for metric_name, pattern in patterns.items():
                match = re.search(pattern, section)
                if match:
                    # Convert to float (or int if it looks like an int)
                    val = match.group(1)
                    try:
                        row_data[metric_name] = float(val) if '.' in val else int(val)
                        found_any = True
                    except ValueError:
                        row_data[metric_name] = val
            
            # Only add the row if we actually found metrics in this chunk
            if found_any:
                extracted_data.append(row_data)

        if not extracted_data:
            print("No evaluation blocks found in the file.")
            return

        # Create DataFrame and save to Excel
        df = pd.DataFrame(extracted_data)
        df.to_csv(output_file, index=False)
        
        print(f"Successfully extracted {len(extracted_data)} evaluation blocks.")
        print(f"Saved to: {output_file}")
        print("\nPreview:")
        print(df.head())

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # You can change these filenames if needed
    INPUT_FILENAME = "./logs/pipeline.out"
    OUTPUT_FILENAME = "./evaluation_metrics.csv"
    
    extract_metrics_to_excel(INPUT_FILENAME, OUTPUT_FILENAME)