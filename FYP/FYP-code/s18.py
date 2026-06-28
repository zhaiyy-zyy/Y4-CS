import pandas as pd
import os

# =========================
# 1. 读取文件
# =========================
file_path = "nq_mitigation_cases_final_all18.csv"

df = pd.read_csv(file_path)
df = df.drop_duplicates(subset=["question"])

print("Total cases:", len(df))
print("Columns:", list(df.columns))


# =========================
# 2. improvement_type 分类 + 比例
# =========================
print("\n=== IMPROVEMENT TYPE DISTRIBUTION ===")

imp_counts = df["improvement_type"].value_counts()
imp_ratio = df["improvement_type"].value_counts(normalize=True)

for k in imp_counts.index:
    print(f"{k}: {imp_counts[k]} ({imp_ratio[k]*100:.2f}%)")


# =========================
# 3. 导出每一类（论文用）
# =========================
os.makedirs("case_groups", exist_ok=True)

for name, group in df.groupby("improvement_type"):
    out_file = f"case_groups/cases_{name}.csv"
    group.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")


# =========================
# 4. FIX / REGRESS / STILL WRONG 核心指标 ⭐
# =========================
total = len(df)

fixed_cnt = len(df[df["improvement_type"] == "fixed"])
regress_cnt = len(df[df["improvement_type"] == "regressed"])
still_wrong_cnt = len(df[df["improvement_type"] == "still_wrong"])

print("\n=== CORE METRICS ===")
print(f"Fix rate: {fixed_cnt/total:.4f}")
print(f"Regress rate: {regress_cnt/total:.4f}")
print(f"Still wrong rate: {still_wrong_cnt/total:.4f}")


# =========================
# 5. 样例展示（论文截图用）
# =========================
def show_samples(df_sub, name):
    print(f"\n=== SAMPLE: {name} ===")
    if len(df_sub) == 0:
        print("None")
        return
    print(df_sub[[
        "question",
        "reference",
        "base_answer",
        "final_answer"
    ]].head(5))


show_samples(df[df["improvement_type"] == "fixed"], "FIXED")
show_samples(df[df["improvement_type"] == "regressed"], "REGRESSED")
show_samples(df[df["improvement_type"] == "still_wrong"], "STILL WRONG")


# =========================
# 6. Relation transition（非常关键‼️）
# =========================
print("\n=== TRANSITION MATRIX ===")

transition = df.groupby(
    ["base_vs_reference", "final_vs_reference"]
).size().reset_index(name="count")

print(transition)

transition.to_csv("transition_matrix.csv", index=False)


# =========================
# 7. action 分布（策略行为）
# =========================
print("\n=== ACTION DISTRIBUTION ===")

action_counts = df["action"].value_counts()
action_ratio = df["action"].value_counts(normalize=True)

for k in action_counts.index:
    print(f"{k}: {action_counts[k]} ({action_ratio[k]*100:.2f}%)")

action_counts.to_csv("action_distribution.csv")


# =========================
# 8. 高质量改进 case_keep ⭐
# =========================
good_cases = df[df["case_keep"] == 1]

print("\n=== GOOD CASES (case_keep=1) ===")
print("Count:", len(good_cases))
print("Ratio:", len(good_cases)/total)

good_cases.to_csv("cases_good_improvements.csv", index=False)


# =========================
# 9. 自动提取所有分类列（通用增强版）
# =========================
print("\n=== AUTO CATEGORICAL SUMMARY ===")

categorical_cols = df.select_dtypes(include=["object"]).columns

summary_rows = []

for col in categorical_cols:
    counts = df[col].value_counts()
    for cat, cnt in counts.items():
        summary_rows.append({
            "column": col,
            "category": cat,
            "count": cnt,
            "ratio": cnt / total
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("all_categorical_summary.csv", index=False)

print("Saved: all_categorical_summary.csv")