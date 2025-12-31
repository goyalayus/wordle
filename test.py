import json
from transformers import AutoTokenizer

# Configuration
MODEL_ID = "Qwen/Qwen2.5-0.5B"
INPUT_FILE = "data/train_sft_145.jsonl"
OUTPUT_FILE = "data/wordle_sft_final.jsonl"
SAFE_LIMIT = 2030  # Leaving 18 tokens for EOS/BOS and template overhead

print(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

clean_count = 0
dropped_count = 0

with open(INPUT_FILE, 'r') as f_in, open(OUTPUT_FILE, 'w') as f_out:
    for line in f_in:
        data = json.loads(line)
        
        # We simulate exactly what the model sees
        text = tokenizer.apply_chat_template(
            data['messages'], 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        token_count = len(tokenizer.encode(text))
        
        if token_count <= SAFE_LIMIT:
            f_out.write(json.dumps(data) + '\n')
            clean_count += 1
        else:
            dropped_count += 1

print("\n" + "="*30)
print(f"Final Dataset: {OUTPUT_FILE}")
print(f"Rows Kept:     {clean_count}")
print(f"Rows Dropped:  {dropped_count}")
print("="*30)
