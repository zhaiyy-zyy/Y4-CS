import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ 读取数据
# ===============================
df = pd.read_csv("truthfulqa_threshold_curve.csv")

plt.figure(figsize=(6,4))

# ===============================
# 2️⃣ 主曲线（统一风格）
# ===============================

# Hallucination
plt.plot(df['t_high'], df['halluc_rate'],
         marker='o', linestyle='-', label="Hallucination")

# Regression（轻微上移避免重叠）
plt.plot(df['t_high'], df['regress_rate'] + 0.002,
         marker='s', linestyle='--', label="Regression")

# ===============================
# 3️⃣ Abstention（TQA已有）
# ===============================
if "abstain_rate" in df.columns:
    plt.plot(df['t_high'], df['abstain_rate'] - 0.002,
             marker='^', linestyle=':', label="Abstention")

# ===============================
# 4️⃣ 图设置（与 FEVER / NQ 完全一致）
# ===============================
plt.xlabel("Threshold (t_high)")
plt.ylabel("Rate")
plt.title("Effect of Threshold on Mitigation (TruthfulQA)")

plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("threshold_analysis_tqa.png", dpi=300)
plt.show()