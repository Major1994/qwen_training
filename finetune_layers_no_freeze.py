import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
import logging

# 设置日志级别和格式
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置 CUDA_VISIBLE_DEVICES，让程序使用第0号到第7号GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# 加载数据集
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir="./hf_cache")

def filter_empty_text(example):
    return example['text'].strip() != ''

# 过滤掉文本为空的样本
filtered_dataset = dataset.filter(filter_empty_text)

# 初始化分词器和模型
model_name = "Qwen3-14B"  # 替换为实际使用的模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 手动创建 device_map
# 假设我们有8个GPU，将24层Transformer均匀分配到这些GPU上
num_gpus = 8

device_map = {"": 0}  # 默认所有未指定的部分放在GPU 0

# 分配嵌入层
device_map["model.embed_tokens"] = 0

num_layers = 967
layers_per_gpu = num_layers // num_gpus
# 分配每一层Transformer
for i in range(num_layers):
    gpu_id = i // layers_per_gpu
    device_map[f"model.layers.{i}"] = f"cuda:{gpu_id}"

# 分配最终的层归一化和输出层
device_map["model.norm"] = f"cuda:{num_gpus - 1}"
device_map["lm_head"] = f"cuda:{num_gpus - 1}"

# 使用 init_empty_weights 创建一个空的模型结构
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)

# 加载权重并分发模型
model = load_checkpoint_and_dispatch(model, checkpoint=model_name, device_map=device_map)

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

# 使用DataCollator进行动态padding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 初始化Accelerator
accelerator = Accelerator()

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results_params_1_8b',
    num_train_epochs=3,
    per_device_train_batch_size=2,  # 根据您的GPU内存调整此值
    gradient_accumulation_steps=8,  # 调整此值以补偿较小的batch size
    save_steps=10_000,
    save_total_limit=2,
    fp16=True,  # 使用混合精度训练加速并节省显存
    prediction_loss_only=True,
    logging_dir='./logs',  # TensorBoard日志目录
    logging_steps=10,  # 每10步记录一次日志
    logging_first_step=True,  # 记录第一步的日志
    report_to=["tensorboard"],  # 启用TensorBoard报告
    evaluation_strategy="steps",  # 可选：定期评估
    eval_steps=1500,  # 每1500步评估一次
)

# 准备Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],  # 使用训练集而不是验证集
    eval_dataset=tokenized_datasets["validation"],  # 可选：如果需要验证集
    data_collator=data_collator,
)

# 使用Accelerator准备训练
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, trainer.optimizer, trainer.train_dataset, trainer.eval_dataset)

# 开始训练
trainer.train()

# 保存微调后的模型
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained('./fine-tuned-model-qwen1_8b')

# 输出训练完成信息
logger.info("Training completed. Model saved to ./fine-tuned-model-qwen1_8b")
