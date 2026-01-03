import pandas as pd
import json
import os

# 1. Setup
INPUT_FILE = 'data/train-00000-of-00001.parquet'
OUTPUT_FILE = 'data/format_tuning_set_clean.jsonl'
TARGET_GAMES = 40 

# --- THE DETAILED RL SYSTEM PROMPT ---
# This matches the logic found in your training data but explains the rules explicitly.
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

print(f"Loading {INPUT_FILE}...")
df = pd.read_parquet(INPUT_FILE)

# 2. Select Reward 1.40 (Long Context Games)
df['reward_rounded'] = df['reward'].round(4)
unique_rewards = sorted(df['reward_rounded'].unique(), reverse=True)
# Target the 3rd tier (1.40) to get longer games with more history
target_reward = unique_rewards[2] 

print(f"Selecting games with Reward: {target_reward}")
long_games_df = df[df['reward_rounded'] == target_reward]

# 3. Sample
if len(long_games_df) < TARGET_GAMES:
    sampled_games = long_games_df
else:
    sampled_games = long_games_df.sample(n=TARGET_GAMES, random_state=42)

# 4. Convert and SWAP System Prompt
sft_dataset = []

for index, row in sampled_games.iterrows():
    # Load original history
    original_history = list(row['prompt']) 
    
    # --- THE SWAP ---
    # Overwrite the old "Competitive player" prompt with our new "Expert AI" prompt
    if original_history[0]['role'] == 'system':
        original_history[0]['content'] = RL_SYSTEM_PROMPT
    else:
        original_history.insert(0, {"role": "system", "content": RL_SYSTEM_PROMPT})
        
    conversation_history = original_history
    
    for turn in row['completion']:
        if turn['role'] == 'assistant':
            # Create the SFT row
            sft_dataset.append({
                "messages": conversation_history + [turn]
            })
            # Add to history for next turn
            conversation_history.append(turn)
        elif turn['role'] == 'user':
            conversation_history.append(turn)

# 5. Save
sft_df = pd.DataFrame(sft_dataset)
sft_df.to_json(OUTPUT_FILE, orient='records', lines=True)

print(f"\nSUCCESS!")
print(f"Generated {len(sft_df)} rows with the DETAILED SYSTEM PROMPT.")
print(f"Dataset saved to: {OUTPUT_FILE}")
