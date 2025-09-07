import transformers
import torch

# 切换为你下载的模型文件目录, 这里的demo是Llama-3-8B-Instruct
# 如果是其他模型，比如qwen，chatglm，请使用其对应的官方demo
model_id = "/media/codingma/LLM/llama3/Meta-Llama-3-8B-Instruct"
model_id = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B"


def model_download():
    #模型下载
    from modelscope import snapshot_download
    # 下载Qwen3-1.7B模型，并打印模型本地存储路径
    # 详细注释：此处调用modelscope的snapshot_download方法，下载Qwen3-1.7B模型（基础版），
    # 并将下载后的本地路径打印出来，方便后续加载和使用。
    qwen3_model_dir = snapshot_download('Qwen/Qwen3-1.7B')
    print(qwen3_model_dir)


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|im_end|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
import pdb;pdb.set_trace()
print(outputs[0]["generated_text"][len(prompt):])