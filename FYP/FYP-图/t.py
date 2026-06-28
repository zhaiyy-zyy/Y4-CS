import pandas as pd
import os

# ==============================
# 1. 读取数据
# ==============================
INPUT_FILE = "truthfulqa_mitigation_cases.csv"
df = pd.read_csv(INPUT_FILE)

print(f"✅ Loaded {len(df)} rows")

# ==============================
# ⭐ 2. 只保留最终 mitigation（关键！！！）
# ==============================
FINAL_POLICY = "detector_conditioned_truthful"

df = df[df["policy"] == FINAL_POLICY].copy()

print(f"🎯 Final policy rows: {len(df)}")

# ==============================
# 3. 数据清理
# ==============================
df["base_gold_ok"] = df["base_gold_ok"].astype(int)
df["final_gold_ok"] = df["final_gold_ok"].astype(int)

df["final_abstain"] = df["final_abstain"].astype(str).str.lower() == "true"

# ==============================
# 4. 计算 improvement_type
# ==============================
df["improvement_type"] = "unchanged_wrong"

df.loc[(df["base_gold_ok"] == 0) & (df["final_gold_ok"] == 1), "improvement_type"] = "improved"
df.loc[(df["base_gold_ok"] == 1) & (df["final_gold_ok"] == 0), "improvement_type"] = "regressed"
df.loc[(df["base_gold_ok"] == 1) & (df["final_gold_ok"] == 1), "improvement_type"] = "unchanged_correct"

# ==============================
# 5. abstain 标记
# ==============================
df["abstain_from_wrong"] = ((df["base_gold_ok"] == 0) & (df["final_abstain"])).astype(int)

# ==============================
# 6. 输出分类文件
# ==============================
OUTPUT_DIR = "tqa_policy_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for cat in df["improvement_type"].unique():
    subset = df[df["improvement_type"] == cat]
    
    output_file = os.path.join(OUTPUT_DIR, f"{cat}.csv")
    subset.to_csv(output_file, index=False)
    
    print(f"📁 Saved {cat}: {len(subset)} rows")

# ==============================
# 7. abstain 单独输出
# ==============================
abstain_df = df[df["final_abstain"] == True]
abstain_df.to_csv(os.path.join(OUTPUT_DIR, "abstain_cases.csv"), index=False)

print(f"📁 Saved abstain_cases: {len(abstain_df)} rows")

# ==============================
# 8. 保存总表
# ==============================
df.to_csv(os.path.join(OUTPUT_DIR, "tqa_final_policy_all.csv"), index=False)

# ==============================
# 9. 打印统计（论文用🔥）
# ==============================
print("\n📊 Improvement Summary:")
print(df["improvement_type"].value_counts(normalize=True))

# ❗这里不用 policy 了（因为已经过滤）
print("\n📊 Counts:")
print(df["improvement_type"].value_counts())

# ==============================
# 10. 示例 case（论文写用）
# ==============================
improved_cases = df[df["improvement_type"] == "improved"]

if len(improved_cases) > 0:
    row = improved_cases.iloc[0]

    print("\n==============================")
    print("=== IMPROVED CASE ===")
    print("Q:", row["question"])

    print("\n--- BASE ---")
    print(row["base_answer"])

    print("\n--- FINAL ---")
    print(row["final_answer"])

    print("\n--- GOLD ---")
    print("Correct:", row["correct_answers"])

    print("\n--- ACTION ---")
    print(row["action"])
else:
    print("⚠️ No improved cases found")