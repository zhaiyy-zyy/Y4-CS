import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读数据
df = pd.read_csv("f_mitigation_cases12.csv")

# 处理 True/False（防止是字符串）
df["base_gold_ok"] = df["base_gold_ok"].astype(str)

correct_scores = df[df["base_gold_ok"] == "True"]["base_ok_score"]
halluc_scores = df[df["base_gold_ok"] == "False"]["base_ok_score"]

# 风格（很关键！）
sns.set(style="whitegrid")

plt.figure(figsize=(7,5))

# 直方图 + KDE（平滑曲线）
sns.histplot(correct_scores, bins=20, kde=True, 
             color="#4C72B0", label="Correct", stat="density", alpha=0.6)

sns.histplot(halluc_scores, bins=20, kde=True, 
             color="#DD8452", label="Hallucinated", stat="density", alpha=0.6)

# 限制范围（关键！）
plt.xlim(0, 1)

# 可选：画 threshold（如果你有）
threshold = 0.20
plt.axvline(x=threshold, linestyle="--", color="black", label="Threshold")

plt.xlabel("Reliability Score", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.title("Score Distribution on FEVER", fontsize=14)

plt.legend()
plt.tight_layout()

# ⭐ 一定先保存再 show
plt.savefig("Score-fever.png", dpi=300)
plt.show()