import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("threshold_curve32.csv")

plt.figure(figsize=(8, 5))

plt.plot(df["t_high"], df["halluc_rate"], marker='o', linewidth=2, label="Hallucination Rate")
plt.plot(df["t_high"], df["regress_rate"], marker='s', linewidth=2, linestyle='--', label="Regression Rate")

plt.xlabel("High Threshold $t_{high}$", fontsize=12)
plt.ylabel("Rate", fontsize=12)
plt.title("Threshold Sensitivity Analysis on Natural Questions", fontsize=13)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("threshold_curve_nq.png", dpi=300)  # 保存论文用
plt.show()