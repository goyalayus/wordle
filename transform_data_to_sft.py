import pandas as pd

# 1. Load the Parquet file
df = pd.read_parquet('data/train-00000-of-00001.parquet')

sft_dataset = []

# 2. Iterate through every game in the original file
for index, row in df.iterrows():
    # Start the history with the initial setup (System Prompt + First User Instruction)
    # We use list() to ensure we create a new copy for every game
    conversation_history = list(row['prompt']) 
    
    # Iterate through the turns in the completion column
    for turn in row['completion']:
        if turn['role'] == 'assistant':
            # --- CREATE A DATASET ROW HERE ---
            # We want the model to see the history and output THIS turn
            
            # Create a full conversation object including the current reply
            current_training_example = {
                "messages": conversation_history + [turn]
            }
            sft_dataset.append(current_training_example)
            
            # Update history so the NEXT row includes this answer
            conversation_history.append(turn)
            
        elif turn['role'] == 'user':
            # If it's the User giving feedback, just add it to history
            # (We don't train on this, we just need it for context)
            conversation_history.append(turn)

# 3. Convert to DataFrame to view or save
sft_df = pd.DataFrame(sft_dataset)

# --- VERIFICATION ---
# Let's print the first 2 rows to prove it works as you asked
print(f"Total training rows generated: {len(sft_df)}")
print("\n=== ROW 1 (Initial Guess) ===")
print(sft_df.iloc[0]['messages'][-1]['content']) # Showing just the response
print("\n=== ROW 2 (Response to Feedback) ===")
print(sft_df.iloc[1]['messages'][-1]['content']) # Showing just the response

# 4. Save to JSONL (Standard format for fine-tuning)
sft_df.to_json("data/my_wordle_sft_data.jsonl", orient='records', lines=True)
print("\nSaved to 'my_wordle_sft_data.jsonl'")
