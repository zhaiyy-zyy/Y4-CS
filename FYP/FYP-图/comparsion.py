import matplotlib.pyplot as plt
import numpy as np

# ===== 正确数据（来自你的实验）=====
datasets = ["FEVER", "NQ", "TruthfulQA"]

base = [0.613, 0.848, 0.857]
proposed = [0.312, 0.805, 0.850]

# Δ（重点！）
delta = [b - p for b, p in zip(base, proposed)]

x = np.arange(len(datasets))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# ===== 柱状图 =====
bars1 = ax.bar(x - width/2, base, width, label="Base")
bars2 = ax.bar(x + width/2, proposed, width, label="Proposed")

# ===== 数值标注 =====
ax.bar_label(bars1, padding=3, fmt="%.3f", fontsize=9)
ax.bar_label(bars2, padding=3, fmt="%.3f", fontsize=9)

# ===== Δ标注（关键提升点）=====
for i in range(len(x)):
    ax.text(
        x[i],
        max(base[i], proposed[i]) + 0.03,
        f"Δ={delta[i]:.3f}",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

# ===== 坐标轴 =====
ax.set_ylabel("Hallucination Rate", fontsize=11)
ax.set_xlabel("Dataset", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(datasets)

# ===== 更合理的范围 =====
ax.set_ylim(0, 1.1)

# ===== 网格（论文必备）=====
ax.grid(axis="y", linestyle="--", alpha=0.4)

# ===== 标题 =====
ax.set_title(
    "Hallucination Rates Before and After Mitigation",
    fontsize=13,
    fontweight="bold"
)

# ===== 图例 =====
ax.legend(frameon=False)

# ===== 保存 =====
plt.tight_layout()
plt.savefig("fig_hallucination_comparison1.png", dpi=300, bbox_inches='tight')

plt.show()