import matplotlib.pyplot as plt
import numpy as np
import os

FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)

datasets = ["TruthfulQA", "FEVER", "Natural Questions"]
means = np.array([0.4039, 0.4061, 0.5175])

plt.figure()
bars = plt.bar(datasets, means)   # ✅ 不要 yerr，就不会有 I 形误差棒
plt.ylabel("Hallucination Rate")
plt.title("Final Hallucination Rate Across Datasets")
plt.ylim(0, 0.6)

# 在柱子上标数值
for bar, value in zip(bars, means):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.01,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "hallucination_rate_datasets.png"), dpi=300)
plt.close()


import matplotlib.pyplot as plt
import numpy as np
import os

FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)

# =========================
# Data
# =========================
runs = ["Run 1", "Run 2", "Run 3"]

tqa_runs = np.array([0.4333, 0.3767, 0.4017])
fever_runs = np.array([0.4000, 0.4017, 0.4167])
nq_runs = np.array([0.5025, 0.5375, 0.5125])

x = np.arange(len(runs))
width = 0.22   # 👈 柱子更瘦

# =========================
# Plot
# =========================
plt.figure(figsize=(10, 6))  # 👈 图整体更宽松

bars1 = plt.bar(x - width, tqa_runs, width, label="TruthfulQA")
bars2 = plt.bar(x, fever_runs, width, label="FEVER")
bars3 = plt.bar(x + width, nq_runs, width, label="Natural Questions")

plt.xticks(x, runs)
plt.ylabel("Hallucination Rate")
plt.title("Hallucination Rate Across Multiple Runs")
plt.ylim(0, 0.65)  # 👈 上方留白
plt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)

# =========================
# Add value labels (looser)
# =========================
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,   # 👈 数字离柱子更远
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=11
        )

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "hallucination_rate_runs.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()