from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen3-32B"  
model = AutoModelForCausalLM.from_pretrained(model_name)

num_layers = 0
for name, params in model.named_paramters():
    print(name)
    num_layers += 1
print(num_layers)
for name, _ in model.named_modules():
    print(name)

