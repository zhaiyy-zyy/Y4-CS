import pandas as pd

# =========================
# 1. 读取文件
# =========================
file_path = "nq_mitigation_cases_final_all15.csv"

df = pd.read_csv(file_path)
df = df.drop_duplicates(subset=["question"])
print("Total cases:", len(df))
print(df.columns)


# =========================
# 2. 按 improvement_type 分类
# =========================
groups = df.groupby("improvement_type")

for name, group in groups:
    print(f"\n=== {name} ===")
    print("Count:", len(group))


# =========================
# 3. 导出每一类（论文用！）
# =========================
for name, group in groups:
    out_file = f"cases_{name}.csv"
    group.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")


# =========================
# 4. 看“真正修复的案例”
# =========================
fixed = df[df["improvement_type"] == "fixed"]

print("\n=== SAMPLE FIXED CASES ===")
print(fixed[[
    "question",
    "reference",
    "base_answer",
    "final_answer"
]].head(5))


# =========================
# 5. 看“变差的案例”（非常重要‼️）
# =========================
regressed = df[df["improvement_type"] == "regressed"]

print("\n=== SAMPLE REGRESSED CASES ===")
print(regressed[[
    "question",
    "reference",
    "base_answer",
    "final_answer"
]].head(5))


# =========================
# 6. 看“仍然错误（hallucination）”
# =========================
still_wrong = df[df["improvement_type"] == "still_wrong"]

print("\n=== SAMPLE STILL WRONG ===")
print(still_wrong[[
    "question",
    "reference",
    "final_answer"
]].head(5))


# =========================
# 7. 按 relation 分析（更高级‼️）
# =========================
rel_groups = df.groupby(["base_vs_reference", "final_vs_reference"])

print("\n=== TRANSITION ANALYSIS ===")
for (b, f), g in rel_groups:
    print(f"{b} -> {f} : {len(g)}")


# =========================
# 8. action 分布（策略分析‼️）
# =========================
print("\n=== ACTION DISTRIBUTION ===")
print(df["action"].value_counts())


# =========================
# 9. case_keep（高质量改进）
# =========================
good_cases = df[df["case_keep"] == 1]

print("\n=== GOOD CASES (case_keep=1) ===")
print("Count:", len(good_cases))

good_cases.to_csv("cases_good_improvements.csv", index=False)