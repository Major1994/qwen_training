import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# 1. 模型名称（请替换为你想使用的 Qwen 模型）
model_name = "Qwen3-32B"  # 或 "Qwen/Qwen-7B", "Qwen/Qwen-14B" 等

# 2. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir="./hf_cache")

# 打印训练集的第一个样本
print("训练集第一个样本：")
print(dataset["train"])

# 打印字段名
print("字段名：", dataset["train"].features)

def filter_empty_text(example):
    return example['text'].strip() != ''

# 过滤掉文本为空的样本
filtered_dataset = dataset.filter(filter_empty_text)

print(f"原始训练集大小: {len(dataset['train'])}")
print(f"过滤后训练集大小: {len(filtered_dataset['train'])}")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=512,
        return_special_tokens_mask=True,
    )

# 对过滤后的数据集进行 tokenization
tokenized_datasets = filtered_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=filtered_dataset["train"].column_names,  # 移除旧列
    desc="Tokenizing and chunking dataset"
)

# 验证第一个样本
sample = tokenized_datasets["train"][0]
print("验证 - input_ids 长度:", len(sample["input_ids"]))
assert len(sample["input_ids"]) > 0, "input_ids 为空！请检查字段名是否正确"
