import pandas as pd

# ================================
# 配置
# ================================
MITI_FILE = "f_mitigation_cases12.csv"
ERROR_FILE = "f_paper_error_cases12.csv"
OUTPUT_FILE = "fever_case_studies.csv"

NUM_CASES_PER_TYPE = 3  # 每类选几个

# ================================
# 读取数据
# ================================
miti_df = pd.read_csv(MITI_FILE)
error_df = pd.read_csv(ERROR_FILE)

print("MITI columns:", miti_df.columns)
print("ERROR columns:", error_df.columns)

# ================================
# 1️⃣ Improved cases
# base错 → final对
# ================================
improved_cases = miti_df[
    (miti_df["base_gold_ok"] == 0) &
    (miti_df["final_gold_ok"] == 1)
]

improved_cases = improved_cases.sample(
    n=min(NUM_CASES_PER_TYPE, len(improved_cases)),
    random_state=42
)

improved_cases["outcome"] = "Improved"

# ================================
# 2️⃣ Regressed cases
# base对 → final错
# ================================
regressed_cases = miti_df[
    (miti_df["base_gold_ok"] == 1) &
    (miti_df["final_gold_ok"] == 0)
]

regressed_cases = regressed_cases.sample(
    n=min(NUM_CASES_PER_TYPE, len(regressed_cases)),
    random_state=42
)

regressed_cases["outcome"] = "Regressed"

# ================================
# 3️⃣ Error cases（原始错误）
# pred错
# ================================
error_cases = error_df[
    error_df["pred_ok"] == 0
]

error_cases = error_cases.sample(
    n=min(NUM_CASES_PER_TYPE, len(error_cases)),
    random_state=42
)

error_cases["outcome"] = "Error"

# ================================
# 统一字段（非常重要）
# ================================
error_cases = error_cases.rename(columns={
    "answer": "base_answer"
})

error_cases["final_answer"] = ""

# ================================
# 合并
# ================================
all_cases = pd.concat([
    improved_cases,
    regressed_cases,
    error_cases
], ignore_index=True)

# ================================
# 去重（避免重复 case）
# ================================
all_cases = all_cases.drop_duplicates(subset=["question"])

# ================================
# 只保留论文需要字段
# ================================
columns_to_keep = [
    "question",
    "base_answer",
    "final_answer",
    "outcome"
]

all_cases = all_cases[[col for col in columns_to_keep if col in all_cases.columns]]

# ================================
# 保存
# ================================
all_cases.to_csv(OUTPUT_FILE, index=False)

print(f"✅ FEVER case studies saved to {OUTPUT_FILE}")