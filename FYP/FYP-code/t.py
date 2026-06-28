import json
from datasets import load_dataset

print("Loading TruthfulQA...")

dataset = load_dataset("truthful_qa", "generation")["validation"]

data_out = []

for i, s in enumerate(dataset):
    item = {
        "id": i,
        "question": s["question"],
        "best_answer": s["best_answer"],
        "correct_answers": s["correct_answers"],
        "incorrect_answers": s["incorrect_answers"]
    }
    data_out.append(item)

# 保存
output_path = "truthfulqa_full.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data_out, f, indent=2, ensure_ascii=False)

print(f"Saved {len(data_out)} samples to {output_path}")