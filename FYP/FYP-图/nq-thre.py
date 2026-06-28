import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ 读取数据
# ===============================
df = pd.read_csv("nq_threshold_curve.csv")

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
# 3️⃣ Abstention（自动处理）
# ===============================
if "abstain_rate" in df.columns:
    abstain = df["abstain_rate"]
elif "valid_answer_rate" in df.columns:
    abstain = 1 - df["valid_answer_rate"]
else:
    abstain = None

if abstain is not None:
    plt.plot(df['t_high'], abstain - 0.002,
             marker='^', linestyle=':', label="Abstention")

# ===============================
# 4️⃣ 图设置（和 FEVER 完全一致）
# ===============================
plt.xlabel("Threshold (t_high)")
plt.ylabel("Rate")
plt.title("Effect of Threshold on Mitigation (NQ)")

plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("threshold_analysis_nq.png", dpi=300)
plt.show()