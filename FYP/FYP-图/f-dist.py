import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===== 🎨 统一风格（和 TQA 一样）=====
plt.rcParams.update({
    "font.size": 11
})
plt.style.use("seaborn-v0_8-whitegrid")

# ===============================
# 1️⃣ 读取数据
# ===============================
df = pd.read_csv("fever_mitigation_cases.csv")

# ===============================
# 2️⃣ 过滤（最终 mitigation）
# ===============================
df = df[df["policy"] == "policy"].copy()
df = df[df["run"] == 0]

# ===============================
# 3️⃣ 清洗
# ===============================
df = df.dropna(subset=[
    "base_ok_score",
    "final_ok_score",
    "base_gold_ok",
    "final_gold_ok"
])

df["base_ok_score"] = df["base_ok_score"].astype(float)
df["final_ok_score"] = df["final_ok_score"].astype(float)

df["base_gold_ok"] = df["base_gold_ok"].astype(bool)
df["final_gold_ok"] = df["final_gold_ok"].astype(bool)

# ===============================
# 4️⃣ question-level 聚合
# ===============================
df_q = df.groupby("question", as_index=False).agg({
    "base_ok_score": "mean",
    "final_ok_score": "mean",
    "base_gold_ok": "max",
    "final_gold_ok": "max"
})

# ===============================
# 5️⃣ outcome（和 TQA 完全统一命名）
# ===============================
def classify(row):
    if (not row["base_gold_ok"]) and row["final_gold_ok"]:
        return "Improved"
    elif row["base_gold_ok"] and (not row["final_gold_ok"]):
        return "Regressed"
    elif row["base_gold_ok"] and row["final_gold_ok"]:
        return "Correct"
    else:
        return "Incorrect"

df_q["outcome"] = df_q.apply(classify, axis=1)

# ===============================
# 6️⃣ 分布数据
# ===============================
correct_scores = df_q[df_q["final_gold_ok"]]["final_ok_score"]
halluc_scores = df_q[~df_q["final_gold_ok"]]["final_ok_score"]

# ===============================
# 7️⃣ 画图（完全对齐 TQA）
# ===============================
fig = plt.figure(figsize=(11,7))

# ===== ① Distribution =====
ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)

bins = np.linspace(0, 1, 25)

ax1.hist(correct_scores, bins=bins, alpha=0.6,
         label="Correct", color="#4CAF50", density=True)

ax1.hist(halluc_scores, bins=bins, alpha=0.6,
         label="Hallucinated", color="#F44336", density=True)

ax1.set_title("Reliability Score Distribution (FEVER)", fontsize=13, fontweight="bold")
ax1.set_xlabel("Reliability Score")
ax1.set_ylabel("Density")
ax1.legend(frameon=False)
ax1.grid(alpha=0.3)

# ===== ② Score Shift =====
ax2 = plt.subplot2grid((2,2), (1,0))

sample_df = df_q.sample(min(300, len(df_q)), random_state=42)

# ⭐ 加 jitter（关键）
jitter = 0.015

x = sample_df["base_ok_score"] + np.random.normal(0, jitter, len(sample_df))
y = sample_df["final_ok_score"] + np.random.normal(0, jitter, len(sample_df))

ax2.scatter(
    x, y,
    alpha=0.5,
    s=25,
    edgecolors='none'
)

ax2.plot([0,1], [0,1], linestyle="--", color="black", linewidth=1)

ax2.set_title("Score Shift (Base → Final)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Base Score")
ax2.set_ylabel("Final Score")
ax2.grid(alpha=0.3)
# ===== ③ Outcome =====
ax3 = plt.subplot2grid((2,2), (1,1))

order = ["Improved", "Correct", "Incorrect", "Regressed"]

counts = df_q["outcome"].value_counts().reindex(order).fillna(0)

colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800"]

bars = ax3.bar(order, counts, color=colors)

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
fig.suptitle("FEVER: Detection and Mitigation Analysis", fontsize=15, fontweight="bold")

# ===============================
# 8️⃣ 保存
# ===============================
plt.tight_layout()
plt.savefig("fever_combined_figure_clean.png", dpi=300, bbox_inches='tight')
plt.show()