import matplotlib.pyplot as plt
import numpy as np

# 数据
datasets = ["FEVER", "NQ", "TruthfulQA"]
base = [0.595, 0.857, 0.883]
policy = [0.295, 0.808, 0.890]

x = np.arange(len(datasets))
width = 0.35

plt.figure(figsize=(6,4))  # ⭐ 论文推荐比例

# bar chart
bars1 = plt.bar(x - width/2, base, width, label="Base")
bars2 = plt.bar(x + width/2, policy, width, label="Proposed")

# 数值标注（publication级）
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                 f"{height:.3f}", ha='center', va='bottom', fontsize=8)

# 轴
plt.xticks(x, datasets, fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Dataset", fontsize=11)
plt.ylabel("Hallucination Rate", fontsize=11)

# 标题（简洁！）
plt.title("Hallucination Reduction Across Datasets", fontsize=11)

# 图例
plt.legend(fontsize=9)

# 保存（关键）
plt.savefig("fig_hallucination_comparison1.png", dpi=300, bbox_inches='tight')

plt.show()