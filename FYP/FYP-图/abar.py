import matplotlib.pyplot as plt
import numpy as np

datasets = [
    ("FEVER",
     ["Base", "Mild", "Strict-K", "Policy"],
     [0.387, 0.425, 0.688, 0.688],
     [0.000, 0.000, 0.000, 0.000]),

    ("NQ",
     ["Base", "Extract-once", "Extract-K", "Policy"],
     [0.102, 0.102, 0.102, 0.125],
     [0.000, 0.000, 0.000, 0.000]),

    ("TruthfulQA",
     ["Base", "Rewrite-once", "Rewrite-K", "Policy"],
     [0.143, 0.148, 0.123, 0.150],
     [0.0067, 0.0367, 0.0050, 0.0050])
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for ax, (title, policies, acc, abstain) in zip(axes, datasets):
    x = np.arange(len(policies))
    width = 0.36

    bars1 = ax.bar(x - width/2, acc, width, label="Accuracy")
    bars2 = ax.bar(x + width/2, abstain, width, label="Abstention")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=25, ha="right")
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.015,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.015,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

axes[0].set_ylabel("Rate", fontsize=11)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, 
           fontsize=12, bbox_to_anchor=(0.5, 1.02))

fig.suptitle("Accuracy and Abstention Across Mitigation Policies", fontsize=15, fontweight="bold", y=1.05)

plt.tight_layout()
plt.savefig("fig2_acc_abstain.png", dpi=300, bbox_inches="tight")
plt.show()