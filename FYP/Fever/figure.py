import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. 读取 CSV 文件
# =========================
file_path = "f_paper_overall_summary11.csv"   # ← 改成你的文件名
df = pd.read_csv(file_path)

print("Data Preview:")
print(df.head())


# =========================
# 2. 基本设置（让图更适合论文）
# =========================
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["font.size"] = 10


# =========================
# 3. 图1：F1 对比（核心）
# =========================
plt.figure()

plt.bar(df["mode"], df["f1_mean"], yerr=df["f1_std"])

plt.xlabel("Mode")
plt.ylabel("F1 Score (mean ± std)")
plt.title("Ablation Study on Detection Signals")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("figure_f1_comparison.png", dpi=300)
plt.show()


# =========================
# 4. 图2：ΔF1（贡献分析🔥）
# =========================
plt.figure()

plt.bar(df["mode"], df["delta_f1_vs_full_mean"])

plt.axhline(0)  # baseline
plt.xlabel("Mode")
plt.ylabel("ΔF1 vs Full Model")
plt.title("Performance Drop Relative to Full Model")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("figure_delta_f1.png", dpi=300)
plt.show()


# =========================
# 5. 图3：Stability（非常重要🔥）
# =========================
plt.figure()

plt.bar(df["mode"], df["f1_std"])

plt.xlabel("Mode")
plt.ylabel("F1 Standard Deviation")
plt.title("Stability Comparison (Lower is Better)")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("figure_stability.png", dpi=300)
plt.show()


# =========================
# 6. 图4：Consistency（高级🔥）
# =========================
x = np.arange(len(df))
width = 0.35

plt.figure()

plt.bar(x, df["question_score_std_mean"], width, label="Question-level")
plt.bar(x + width, df["answer_score_std_mean"], width, label="Answer-level")

plt.xticks(x + width / 2, df["mode"], rotation=30)
plt.xlabel("Mode")
plt.ylabel("Score Variance")
plt.title("Consistency Across Modes")
plt.legend()

plt.tight_layout()
plt.savefig("figure_consistency.png", dpi=300)
plt.show()


# =========================
# 7.（可选）打印总结
# =========================
print("\nSummary:")
print(df[["mode", "f1_mean", "f1_std", "delta_f1_vs_full_mean"]])