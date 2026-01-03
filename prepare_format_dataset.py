import pandas as pd
import numpy as np
import os

# 1. Setup
INPUT_FILE = 'data/train-00000-of-00001.parquet'
OUTPUT_FILE = 'data/format_tuning_set.jsonl'
TARGET_GAMES = 40  # 40 games * ~5 turns = ~200 training rows

print(f"Loading {INPUT_FILE}...")
df = pd.read_parquet(INPUT_FILE)

# 2. Analyze Rewards to find the "Second Last" or "Long Game" tier
# We round to avoid float precision issues
df['reward_rounded'] = df['reward'].round(4)
unique_rewards = sorted(df['reward_rounded'].unique(), reverse=True)

print("\nAvailable Reward Tiers (Score | Games):")
for r in unique_rewards:
    count = len(df[df['reward_rounded'] == r])
    print(f"  {r:.4f}  |  {count}")

# 3. Select the target tier
# unique_rewards[0] is Best (1.53 - Short)
# unique_rewards[2] is usually 1.40 (Longer)
# unique_rewards[3] is usually 1.36 (Longest/Hardest)
target_reward = unique_rewards[2] # Selecting 1.4000 (The "Second Last" tier)

print(f"\nSelecting games with Reward: {target_reward}")
long_games_df = df[df['reward_rounded'] == target_reward]

# 4. Sample the specific number of games
if len(long_games_df) < TARGET_GAMES:
    print(f"Warning: Only {len(long_games_df)} games available. Using all.")
    sampled_games = long_games_df
else:
    sampled_games = long_games_df.sample(n=TARGET_GAMES, random_state=42)

# 5. Convert to SFT Format (Conversational)
sft_dataset = []

for index, row in sampled_games.iterrows():
    # Start with system prompt + first user message
    conversation_history = list(row['prompt']) 
    
    for turn in row['completion']:
        if turn['role'] == 'assistant':
            # Create a training row: History -> Target Response
            # This captures the "Context Length" perfectly
            sft_dataset.append({
                "messages": conversation_history + [turn]
            })
            
            # Update history so the NEXT row has even more context
            conversation_history.append(turn)
            
        elif turn['role'] == 'user':
            conversation_history.append(turn)

# 6. Save
sft_df = pd.DataFrame(sft_dataset)
sft_df.to_json(OUTPUT_FILE, orient='records', lines=True)

print(f"\nSUCCESS!")
print(f"Generated {len(sft_df)} training rows from {len(sampled_games)} long games.")
print(f"Dataset saved to: {OUTPUT_FILE}")
print("Use this file in the Unsloth Alpaca Notebook.")
