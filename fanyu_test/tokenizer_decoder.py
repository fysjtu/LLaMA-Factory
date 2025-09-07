from transformers import AutoTokenizer
from constants import MODEL_ID
# 假设使用的是Qwen3-1.7B模型
model_name = MODEL_ID
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 包含特殊token的ID序列
# 编码 "<|im_start|>user\nHello<|im_end|>"
token_ids = [151644, 8948, 198, 9953, 151645] 

# 默认行为：不跳过特殊tokens
decoded_text_with_specials = tokenizer.decode(token_ids, skip_special_tokens=False)
print(f"With special tokens: '{decoded_text_with_specials}'")
# 输出: With special tokens: '<|im_start|>user\nHello<|im_end|>'

# 设置为True：跳过特殊tokens
decoded_text_clean = tokenizer.decode(token_ids, skip_special_tokens=True)
print(f"Without special tokens: '{decoded_text_clean}'")
# 输出: Without special tokens: 'user\nHello'