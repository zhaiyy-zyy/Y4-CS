import pandas as pd
import matplotlib.pyplot as plt

# 读取你的threshold分析文件
df = pd.read_csv("fever_threshold_analysis.csv")  # 改成你的文件名

plt.figure(figsize=(6,4))

# Hallucination
plt.plot(df['t_high'], df['halluc_rate'],
         marker='o', linestyle='-', label="Hallucination")

# Regression（稍微上移）
plt.plot(df['t_high'], df['regress_rate'] + 0.002,
         marker='s', linestyle='--', label="Regression")

# Abstention（稍微下移）
plt.plot(df['t_high'], df['abstain_rate'] - 0.002,
         marker='^', linestyle=':', label="Abstention")

plt.xlabel("Threshold (t_high)")
plt.ylabel("Rate")
plt.title("Effect of Threshold on Mitigation (FEVER)")

plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("threshold_analysis-f.png", dpi=300)
plt.show()