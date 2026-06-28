from datasets import load_dataset
import numpy as np

print("Loading dataset...")
dataset = load_dataset("natural_questions", split="validation")

print("Sampling...")
np.random.seed(42)
indices = np.random.choice(len(dataset), size=1000, replace=False)

subset = dataset.select(indices)

print("Saving...")
subset.to_json("nq_subset.json")

print("Done!")