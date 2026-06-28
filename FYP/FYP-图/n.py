import pandas as pd
import os

# ==============================
# 1️⃣ 读取数据
# ==============================
INPUT_FILE = "nq_mitigation_cases.csv"
OUTPUT_DIR = "nq_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

print("原始数据行数:", len(df))
print("列名:", df.columns)   # ⭐建议保留

# ==============================
# 2️⃣ 提取最终 mitigation
# ==============================
policy_name = "detector_conditioned_extract"

policy_df = df[df["policy"] == policy_name].copy()

print("最终 mitigation 行数:", len(policy_df))

policy_df.to_csv(os.path.join(OUTPUT_DIR, "nq_final_policy.csv"), index=False)

# ==============================
# 3️⃣ 分类（修正版！！！）
# ==============================
def classify(row):
    base = row["base_gold_ok"]
    final = row["final_gold_ok"]

    if base == 0 and final == 1:
        return "Improved"
    elif base == 1 and final == 1:
        return "Correct (unchanged)"
    elif base == 1 and final == 0:
        return "Regressed"
    else:
        return "Incorrect (unchanged)"

policy_df["outcome"] = policy_df.apply(classify, axis=1)

# ==============================
# 4️⃣ 输出统计
# ==============================
print("\n分类统计：")
print(policy_df["outcome"].value_counts())

# ==============================
# 5️⃣ 分类保存
# ==============================
for outcome in policy_df["outcome"].unique():
    subset = policy_df[policy_df["outcome"] == outcome]

    filename = f"nq_{outcome.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
    subset.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

    print(f"{outcome}: {len(subset)} 条")

# ==============================
# 6️⃣ Case study（论文用）
# ==============================
case_df = policy_df.groupby("outcome").head(5)

case_df.to_csv(os.path.join(OUTPUT_DIR, "nq_case_studies.csv"), index=False)

print("\n已生成 case study 文件")