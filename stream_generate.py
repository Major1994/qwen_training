from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# 模型名称
model_id = "Qwen3-8B"  # 注意：需确保模型路径正确（本地或 Hugging Face）

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)

# 定义输入消息（对话格式）
messages = [
    {"role": "user", "content": "阿里巴巴的竹希是谁。500字"}
]

# 直接生成 tokenized 输入（无需手动转换为字符串）
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,  # 添加生成提示（如 "请回答："）
    return_tensors="pt"          # 直接返回 tokenized 结果
).to(model.device)

# 初始化流式输出器
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 生成文本并流式输出
outputs = model.generate(
    inputs,
    max_new_tokens=100000000,         # 控制生成长度（可调整）
    temperature=0.2,            # 控制随机性
    do_sample=True,             # 启用采样（非贪心搜索）
    streamer=streamer,          # 启用流式输出
    pad_token_id=tokenizer.eos_token_id  # 避免 pad_token 警告
)
