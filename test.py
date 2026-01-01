import json
from transformers import AutoTokenizer

# 1. SETUP - Targeting Qwen 3
# Note: As of late 2025/early 2026, the ID is Qwen/Qwen3-0.6B
model_id = "Qwen/Qwen3-0.6B"
input_file = "data/train_sft_145.jsonl"
SAFE_LIMIT = 2030 

print(f"Loading Qwen 3 Tokenizer: {model_id}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    # Fallback to 2.5 if 3 isn't available in your cache yet
    print("Trying fallback or check your model ID.")
    exit()

lengths = []
over_limit_count = 0

print(f"Analyzing {input_file} for Qwen 3 compatibility...")

with open(input_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        messages = data['messages']
        
        # Apply the Qwen 3 chat template
        # This is critical because Qwen 3 adds specific reasoning/control tokens
        full_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        token_count = len(tokenizer.encode(full_text))
        lengths.append(token_count)
        
        if token_count > SAFE_LIMIT:
            over_limit_count += 1

# 3. REPORT STATS
max_len = max(lengths)
avg_len = sum(lengths) / len(lengths)

print("\n" + "="*40)
print("       QWEN 3 TOKEN ANALYSIS")
print("="*40)
print(f"Model ID:          {model_id}")
print(f"Total SFT Rows:    {len(lengths)}")
print(f"Max Seq Length:    {max_len} tokens")
print(f"Avg Seq Length:    {int(avg_len)} tokens")
print(f"Rows > {SAFE_LIMIT}:     {over_limit_count}")
print("-" * 40)

if over_limit_count == 0:
    print(f"✅ PERFECT: All rows fit within your {SAFE_LIMIT} limit.")
else:
    percentage = (over_limit_count / len(lengths)) * 100
    print(f"⚠️ WARNING: {over_limit_count} rows ({percentage:.2f}%) exceed the limit.")
    print(f"You should re-run the clean_data script for Qwen 3.")
print("="*40)
