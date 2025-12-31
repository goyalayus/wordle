import pandas as pd
import requests
import random
import json
import os

# 1. Configuration
TRAIN_DATA_PATH = 'data/train-00000-of-00001.parquet'
OUTPUT_PATH = 'data/eval_words.json'
# URL for the original 2,315 Wordle solutions (alphabetical)
WORDLE_URL = "https://gist.githubusercontent.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b/raw/wordle-answers-alphabetical.txt"

def build_eval_set():
    # 2. Download the official Wordle target word list
    print("Fetching official Wordle answer list...")
    response = requests.get(WORDLE_URL)
    if response.status_code != 200:
        print("Error: Could not download word list.")
        return
    
    master_list = set(response.text.splitlines())
    print(f"Total Master Words: {len(master_list)}")

    # 3. Load the training data to find words to exclude
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"Error: {TRAIN_DATA_PATH} not found.")
        return
    
    df = pd.read_parquet(TRAIN_DATA_PATH)
    # Get unique words used as 'answer' in training
    train_answers = set(df['answer'].unique())
    print(f"Words to exclude (from training): {len(train_answers)}")

    # 4. Filter: Master List - Training Answers
    candidate_words = list(master_list - train_answers)
    print(f"Remaining candidates for evaluation: {len(candidate_words)}")

    # 5. Sample 100 words
    if len(candidate_words) < 100:
        print("Warning: Less than 100 unique candidates found. Using all available.")
        eval_set = candidate_words
    else:
        eval_set = random.sample(candidate_words, 100)

    # 6. Save to JSON
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(eval_set, f, indent=4)
    
    print(f"\nSuccess! 100 evaluation words saved to {OUTPUT_PATH}")
    print(f"Example words: {eval_set[:5]}")

if __name__ == "__main__":
    build_eval_set()
