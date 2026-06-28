import matplotlib.pyplot as plt
import numpy as np

modes = ["full", "no_judge", "no_mnli", "no_selfcheck", "judge_only"]

# ===== 3 datasets =====
fever_means = [0.9554, 0.7474, 0.9554, 0.9554, 0.9554]
fever_stds  = [0.0027, 0.0224, 0.0027, 0.0027, 0.0027]

nq_means = [0.9760, 0.9736, 0.9830, 0.9756, 0.9486]
nq_stds  = [0.0060, 0.0073, 0.0070, 0.0027, 0.0046]

tqa_means = [0.9342, 0.9352, 0.9355, 0.9360, 0.9360]
tqa_stds  = [0.0006, 0.0005, 0.0005, 0.0002, 0.0002]

datasets = [
    ("FEVER", fever_means, fever_stds),
    ("NQ", nq_means, nq_stds),
    ("TruthfulQA", tqa_means, tqa_stds)
]

# ===== 3 subplots =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, means, stds) in enumerate(datasets):
    ax = axes[i]

    bars = ax.bar(modes, means, yerr=stds, capsize=5)

    ax.set_title(name)
    ax.set_ylim(0.7, 1.0)
    ax.set_xlabel("Model Variant")
    if i == 0:
        ax.set_ylabel("F1 Score")

    # ⭐ 标注数值
    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y, f"{y:.3f}",
                ha='center', va='bottom', fontsize=9)

# 布局优化
plt.tight_layout()

# ⭐ 保存论文图
plt.savefig("fig_ablation.png", dpi=300, bbox_inches='tight')

plt.show()