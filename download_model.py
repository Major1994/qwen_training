from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载预训练模型和分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B", device_map="auto", trust_remote_code=True).eval()

# 设定您想要保存模型和分词器的路径
save_directory = "./my_qwen7b_model"

# 保存分词器
tokenizer.save_pretrained(save_directory)

# 保存模型
model.save_pretrained(save_directory)

print(f"模型和分词器已成功保存至 {save_directory}")
