import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset
import json
import os

def prepare_datasets():
    print("--- Loading Datasets ---")
    
    # 1. Alpaca Cleaned (English)
    try:
        alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
        print(f"Loaded Alpaca Cleaned: {len(alpaca)} samples")
    except Exception as e:
        print(f"Error loading Alpaca: {e}")
        alpaca = Dataset.from_list([])
    
    # 2. German Tasks
    try:
        german = load_dataset("philschmid/translated_tasks_de_google_52k", split="train")
        print(f"Loaded German Tasks: {len(german)} samples")
    except Exception as e:
        print(f"Error loading German tasks: {e}")
        german = Dataset.from_list([])
    
    # 3. OpenAssistant (Multilingual)
    try:
        oasst = load_dataset("OpenAssistant/oasst2", split="train")
        # OASST2 has a tree structure, we need to flatten it to instruction-output pairs
        # For simplicity, we filter for English and German conversations
        oasst_df = oasst.to_pandas()
        oasst_filtered = oasst_df[oasst_df["lang"].isin(["en", "de"])]
        
        # Helper to convert OASST to Alpaca format
        oasst_data = []
        for _, row in oasst_filtered.iterrows():
            if row["role"] == "prompter":
                # Find the next message which should be the assistant's response
                responses = oasst_df[oasst_df["parent_id"] == row["message_id"]]
                for _, resp in responses.iterrows():
                    if resp["role"] == "assistant":
                        oasst_data.append({
                            "instruction": row["text"],
                            "input": "",
                            "output": resp["text"]
                        })
        oasst_ds = Dataset.from_list(oasst_data)
        print(f"Loaded OASST2 (En/De): {len(oasst_ds)} samples")
    except Exception as e:
        print(f"Error loading OASST2: {e}")
        oasst_ds = Dataset.from_list([])

    # 4. Custom Vietnamese Mix (Grammar/Pronunciation Correction)
    # Load data from JSON file
    try:
        json_path = os.path.abspath("vn_mix_data.json")
        with open(json_path, "r", encoding="utf-8") as f:
            vn_mix_data = json.load(f)
        vn_ds = Dataset.from_list(vn_mix_data)
        print(f"Loaded Vietnamese Mix Data: {len(vn_ds)} samples")
    except FileNotFoundError:
        print("Warning: vn_mix_data.json not found. Skipping Vietnamese mix data.")
        vn_ds = Dataset.from_list([]) 
    except json.JSONDecodeError:
        print("Error: vn_mix_data.json is invalid JSON. Skipping.")
        vn_ds = Dataset.from_list([])

    print("--- Merging and Formatting ---")
    # Combine all
    datasets_to_concat = [d for d in [alpaca, german, oasst_ds, vn_ds] if len(d) > 0]
    if not datasets_to_concat:
        print("No datasets loaded!")
        return

    combined_dataset = concatenate_datasets(datasets_to_concat)
    
    # Shuffle
    combined_dataset = combined_dataset.shuffle(seed=42)
    
    # Save to disk
    output_path = "./processed_dataset"
    combined_dataset.save_to_disk(output_path)
    print(f"Dataset prepared and saved to {output_path} with {len(combined_dataset)} samples.")

if __name__ == "__main__":
    prepare_datasets()
