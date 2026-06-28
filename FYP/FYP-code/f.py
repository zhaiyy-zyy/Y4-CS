from datasets import load_dataset
import numpy as np

print("Loading FEVER dataset...")

dataset = load_dataset("fever", "v1.0", split="train")

print("Grouping by label...")

# 分组
label_groups = {
    "SUPPORTS": [],
    "REFUTES": [],
    "NOT ENOUGH INFO": []
}

for i, item in enumerate(dataset):
    label = item["label"]
    if label in label_groups:
        label_groups[label].append(i)

# 每类取多少
N_PER_CLASS = 1000  # 1000 × 3 = 3000

np.random.seed(42)

indices = []

for label, idxs in label_groups.items():
    sampled = np.random.choice(idxs, size=N_PER_CLASS, replace=False)
    indices.extend(sampled)

# 打乱
np.random.shuffle(indices)

subset = dataset.select(indices)

print("Saving JSON...")
subset.to_json("fever_subset.json")

print("Done!")