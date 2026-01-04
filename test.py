import pandas as pd
import json
import os

# 1. Paths - Corrected for your local wordle directory
input_parquet = 'data/train-00000-of-00001.parquet'
output_jsonl = 'data_full_experiment.jsonl'

# --- THE DETAILED RL SYSTEM PROMPT ---
RL_SYSTEM_PROMPT = (
    "You are an expert AI playing Wordle.\n"
    "GOAL: Guess the secret 5-letter word in 6 tries.\n\n"
    "GAME RULES:\n"
    "1. You must input a valid 5-letter English word.\n"
    "2. Feedback is given for each letter:\n"
    "   - G (Green): The letter is in the word and in the CORRECT position.\n"
    "   - Y (Yellow): The letter is in the word but in the WRONG position.\n"
    "   - X (Gray): The letter is NOT in the word (or no extra copies exist).\n\n"
    "LOGIC & STRATEGY:\n"
    "- Eliminate Impossible Letters: Never use a letter marked 'X' again.\n"
    "- Lock in Knowns: If a letter is 'G', keep it in that exact spot.\n"
    "- Move Yellows: If a letter is 'Y', you must use it, but try a different spot.\n\n"
    "FORMATTING:\n"
    "First, think step-by-step inside <think>...</think> tags about which letters are valid.\n"
    "Then, output your guess inside <guess>[word]</guess> tags."
)

# 2. Load the full 1,000 games
if not os.path.exists(input_parquet):
    print(f"Error: Could not find {input_parquet}")
    exit()

print(f"Loading {input_parquet}...")
df = pd.read_parquet(input_parquet)
print(f"Loaded {len(df)} games.")

sft_dataset = []

# 3. Transform every game into SFT turns
for index, row in df.iterrows():
    original_history = list(row['prompt']) 
    
    # Overwrite with the detailed prompt
    if original_history[0]['role'] == 'system':
        original_history[0]['content'] = RL_SYSTEM_PROMPT
    
    conversation_history = original_history
    
    for turn in row['completion']:
        if turn['role'] == 'assistant':
            sft_dataset.append({"messages": conversation_history + [turn]})
            conversation_history.append(turn)
        elif turn['role'] == 'user':
            conversation_history.append(turn)

# 4. Save
with open(output_jsonl, 'w') as f:
    for entry in sft_dataset:
        f.write(json.dumps(entry) + '\n')

print(f"\nSUCCESS! Generated {len(sft_dataset)} training rows.")
print(f"Dataset saved to: {output_jsonl}")
