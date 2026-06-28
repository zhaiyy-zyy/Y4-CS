import pandas as pd

# ================================
# 配置
# ================================
MITI_FILE = "nq_mitigation_cases_final_all30.csv"
ERROR_FILE = "nq_paper_error_cases_final_all30.csv"

OUTPUT_FILE = "selected_case_studies.csv"

NUM_IMPROVED = 5
NUM_REGRESSED = 3
NUM_ERROR = 30

# ================================
# 读取数据
# ================================
miti_df = pd.read_csv(MITI_FILE)
error_df = pd.read_csv(ERROR_FILE)

print("MITI columns:", miti_df.columns)
print("ERROR columns:", error_df.columns)

# ================================
# 1️⃣ Improved（选提升最大的）
# ================================
improved_cases = miti_df[
    miti_df["delta_ok_score"] > 0
].sort_values(by="delta_ok_score", ascending=False).head(NUM_IMPROVED)

improved_cases["outcome"] = "Improved"

# ================================
# 2️⃣ Regressed（选下降最明显的）
# ================================
regressed_cases = miti_df[
    miti_df["delta_ok_score"] < 0
].sort_values(by="delta_ok_score", ascending=True).head(NUM_REGRESSED)

regressed_cases["outcome"] = "Regressed"

# ================================
# 3️⃣ Error（选最不确定/最差）
# ================================
error_cases = error_df.sort_values(by="score").head(NUM_ERROR)

error_cases["outcome"] = "Error"

# ================================
# 4️⃣ 字段统一（非常关键）
# ================================
error_cases = error_cases.rename(columns={
    "answer": "base_answer",
    "reference": "gold_answer"
})

# ================================
# 5️⃣ 合并
# ================================
all_cases = pd.concat([
    improved_cases,
    regressed_cases,
    error_cases
], ignore_index=True)

# ================================
# 6️⃣ 只保留论文字段
# ================================
columns_to_keep = [
    "question",
    "base_answer",
    "final_answer",
    "reference",   # 你这里是 reference
    "outcome"
]

# rename reference → gold_answer
all_cases = all_cases.rename(columns={"reference": "gold_answer"})

all_cases = all_cases[[col for col in columns_to_keep if col in all_cases.columns]]

# ================================
# 7️⃣ 保存
# ================================
all_cases.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Saved {len(all_cases)} cases to {OUTPUT_FILE}")