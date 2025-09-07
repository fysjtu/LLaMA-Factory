from transformers import pipeline

model_id = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B"


def pp_clf():
    # 指定模型；可用本地目录或 Hugging Face 仓库名
    clf = pipeline(
        task="text-classification",
        model=model_id,    
        top_k=None,          # None 返回最高分一个；设为整数可返回 Top-K
        truncation=True,     # 长文本截断
        device_map="auto"    # 自动放到可用 GPU 上
    )
    res = clf(["I love it.", "This is bad."])
    # [{'label': 'POSITIVE', 'score': ...}, {'label': 'NEGATIVE', 'score': ...}]
    import pdb;
    pdb.set_trace()


def pp_gen():
    gen = pipeline(
        "text-generation",
        # model="gpt2",
        model=model_id,    
        model_kwargs={"torch_dtype": torch.float16},  # 降精度提速（GPU）
        device_map="auto"
    )
    out = gen(
        "Once upon a time",
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        num_return_sequences=1,
        eos_token_id=None  # 可不传；需要多终止符可传列表（均为 int）
    )
    print(out[0]["generated_text"])
    import pdb;
    pdb.set_trace()


def pp_gen_chat():
    pipe = pipeline(
        "text-generation",
        model=model_id,    
        device_map="auto"
    )

    messages = [
        {"role": "system", "content": "你是乐于助人的助手。"},
        {"role": "user", "content": "用一句话解释量子纠缠。"}
    ]

    # 使用 tokenizer 的聊天模板，将 messages 转为字符串 prompt
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = pipe(
        prompt,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        # 避免 None：仅在你确认为 int 或非空列表时传 eos_token_id
    )
    print(outputs[0]["generated_text"][len(prompt):])
    import pdb;
    pdb.set_trace()


def pp_ner():
    ner = pipeline("ner", 
        # model="dslim/bert-base-NER", 
        model=model_id,    
        aggregation_strategy="simple")
    # aggregation_strategy="simple" 合并子词为实体片段
    print(ner("My name is Sarah and I live in London."))
    import pdb;
    pdb.set_trace()


def pp_qa():
    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    print(qa({"question": "Where do I live?", "context": "My name is Sarah and I live in London."}))
    import pdb;
    pdb.set_trace()


def pp_sum():
    summ = pipeline("summarization", model="facebook/bart-large-cnn", device_map="auto")
    print(summ("长文...", max_length=120, min_length=30, do_sample=False))
    import pdb;
    pdb.set_trace()


def pp_trans():
    trans = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
    print(trans("Transformers makes NLP easy."))
    import pdb;
    pdb.set_trace()


def pp_fe():
    fe = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
    emb = fe("hello world", return_tensors=True)  # 返回 PyTorch/Numpy 张量
    # shape: [batch, seq_len, hidden_size]    
    import pdb;
    pdb.set_trace()


if __name__ == "__main__":
    pp_clf()
    pp_gen()
    pp_gen_chat()
    pp_ner()
    pp_qa()
    pp_sum()
    pp_trans()
    pp_fe()