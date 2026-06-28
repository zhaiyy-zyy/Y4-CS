import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===== 1️⃣ 读取数据 =====
df = pd.read_csv("truthfulqa_mitigation_cases.csv")
# ⭐ 只保留最终 mitigation（很重要）
df = df[df["policy"] == "detector_conditioned_truthful"].copy()
df = df[df["run"] == 0]
# ⭐ 每个 question 只保留一次
df = df.drop_duplicates(subset=["question"])

# ===== 2️⃣ 分数分布 =====
correct_scores = df[df["final_gold_ok"] == 1]["final_ok_score"]
halluc_scores = df[df["final_gold_ok"] == 0]["final_ok_score"]

# ===== 3️⃣ outcome 分类 =====
def classify(row):
    if row["base_gold_ok"] == 0 and row["final_gold_ok"] == 1:
        return "Improved"
    elif row["base_gold_ok"] == 1 and row["final_gold_ok"] == 0:
        return "Regressed"
    elif row["base_gold_ok"] == 1 and row["final_gold_ok"] == 1:
        return "Correct"
    elif row["final_abstain"] == True:
        return "Abstained"
    else:
        return "Incorrect"

df["outcome"] = df.apply(classify, axis=1)

# ===== 🎨 统一字体大小 =====
plt.rcParams.update({
    "font.size": 11
})

# ===== 4️⃣ 画图 =====
fig = plt.figure(figsize=(11,7))

# ===== 1️⃣ Distribution =====
ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)

bins = np.linspace(0, 1, 25)

ax1.hist(correct_scores, bins=bins, alpha=0.6, label="Correct", color="#4CAF50", density=True)
ax1.hist(halluc_scores, bins=bins, alpha=0.6, label="Hallucinated", color="#F44336", density=True)

ax1.set_title("Reliability Score Distribution (TruthfulQA)", fontsize=13, fontweight="bold")
ax1.set_xlabel("Reliability Score")
ax1.set_ylabel("Density")
ax1.legend(frameon=False)
ax1.grid(alpha=0.3)

# ===== 2️⃣ Score shift =====
ax2 = plt.subplot2grid((2,2), (1,0))

# ⭐ 减少点数（避免太密）
sample_df = df.sample(min(300, len(df)))

ax2.scatter(
    sample_df["base_ok_score"],
    sample_df["final_ok_score"],
    alpha=0.5,
    s=20,
    edgecolors='none'
)

ax2.plot([0,1], [0,1], linestyle="--", color="black", linewidth=1)

ax2.set_title("Score Shift (Base → Final)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Base Score")
ax2.set_ylabel("Final Score")
ax2.grid(alpha=0.3)

# ===== 3️⃣ Outcome bar =====
ax3 = plt.subplot2grid((2,2), (1,1))

order = ["Improved", "Correct", "Incorrect", "Regressed", "Abstained"]

counts = df["outcome"].value_counts().reindex(order).fillna(0)

colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9E9E9E"]

bars = ax3.bar(order, counts, color=colors)

# ⭐ 数值标注
for bar in bars:
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width()/2,
        height + 2,
        f"{int(height)}",
        ha="center",
        fontsize=9
    )

ax3.set_title("Mitigation Outcomes", fontsize=12, fontweight="bold")
ax3.set_ylabel("Count")
ax3.tick_params(axis='x', rotation=20)
ax3.grid(axis='y', alpha=0.3)

# ===== 总标题 =====
fig.suptitle("TruthfulQA: Detection and Mitigation Analysis", fontsize=15, fontweight="bold")

plt.tight_layout()
plt.savefig("tqa_combined_figure_clean.png", dpi=300, bbox_inches='tight')
plt.show()