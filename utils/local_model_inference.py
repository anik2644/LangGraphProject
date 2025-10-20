from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./models/gemma-2b"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu",          # 👈 run on CPU
    torch_dtype=torch.float32  # full precision to avoid NaN
)

prompts = [
    "Explain what is photosynthesis in one sentence.",
    "Translate English to French: I love learning.",
    "Summarize: The quick brown fox jumps over the lazy dog.",
    "Answer this question: What is the capital of Japan?",
    "Write a short poem about friendship."
]

for i, prompt in enumerate(prompts, 1):
    print(f"\n=== Prompt {i} ===")
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("➡️ Output:", result)
