import json
from transformers import AutoTokenizer

# 1. SETUP
# Using Qwen2.5-0.5B tokenizer for accurate counting
model_id = "Qwen/Qwen2.5-0.5B"
input_file = "data/train_sft_145.jsonl"

print(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

lengths = []

print(f"Analyzing {input_file}...")

# 2. READ JSONL AND TOKENIZE
with open(input_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        messages = data['messages']
        
        # Apply the Qwen chat template to match exact training format
        # tokenize=False gives us the string, then we encode to count tokens
        full_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        token_count = len(tokenizer.encode(full_text))
        lengths.append(token_count)

# 3. REPORT STATS
max_len = max(lengths)
avg_len = sum(lengths) / len(lengths)

print("\n" + "="*40)
print("       TOKEN LENGTH ANALYSIS")
print("="*40)
print(f"Total SFT Rows:    {len(lengths)}")
print(f"Max Seq Length:    {max_len} tokens")
print(f"Avg Seq Length:    {int(avg_len)} tokens")
print("-" * 40)

# 4. VRAM PREDICTION PREP
if max_len > 1024:
    print(f"NOTE: Your longest example is {max_len} tokens.")
    print("This will be the primary driver of VRAM usage during QLoRA.")
else:
    print("Sequence lengths are short. This is very good for low-VRAM GPUs.")
print("="*40)
