import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# 1. 模型名称（请替换为你想使用的 Qwen 模型）
model_name = "Qwen3-32B"  # 或 "Qwen/Qwen-7B", "Qwen/Qwen-14B" 等

# 2. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Qwen 使用 eos_token 作为 pad_token

# 3. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,  # 使用半精度加速训练
)

print("原始模型结构：")
print(model)

# 4. 配置 LoRA
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
]

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 查看可训练参数量（通常会显著减少）

# 5. 加载并处理数据集
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir="./hf_cache")

def filter_empty_text(example):
    return example['text'].strip() != ''

# 过滤掉文本为空的样本
filtered_dataset = dataset.filter(filter_empty_text)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,  # 动态 padding 在 data collator 中处理
        max_length=512,
        return_special_tokens_mask=True,
    )

tokenized_datasets = filtered_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing and chunking dataset"
)
print(tokenized_datasets["train"][0])

# 6. 数据整理器（用于动态 padding 和语言建模任务）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 因为是 causal LM（自回归），不是 masked LM
)

# 7. 训练参数
training_args = TrainingArguments(
    output_dir="./qwen-lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # 根据 GPU 显存调整
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
    bf16=False,  # 如果支持 bfloat16 可开启（如 A100/H100）
    fp16=True,   # 使用 float16
    optim="adamw_torch",  # 推荐
    report_to="tensorboard",  # 使用 tensorboard
    remove_unused_columns=False,  # PEFT 有时需要保留
    gradient_checkpointing=False,  # 节省显存（可选）
)

# 8. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 9. 开始训练
trainer.train()

# 10. 保存最终模型（LoRA 权重）
trainer.save_model("./qwen-lora-final")
print("模型已保存到 ./qwen-lora-final")


