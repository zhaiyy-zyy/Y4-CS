import pandas as pd

# ================================
# 配置
# ================================
MITI_FILE = "truthfulqa_mitigation_cases_all19.csv"
ERROR_FILE = "truthfulqa_paper_error_cases_all19.csv"

OUTPUT_FILE = "tqa_case_studies.csv"

NUM_CASES_PER_TYPE = 3
SEED = 42

# ================================
# 读取数据
# ================================
miti_df = pd.read_csv(MITI_FILE)
error_df = pd.read_csv(ERROR_FILE)

print("MITI columns:", miti_df.columns)
print("ERROR columns:", error_df.columns)


# ================================
# 1️⃣ Improved（base错 → final对）
# ================================
improved = miti_df[
    (miti_df["base_gold_ok"] == 0) &
    (miti_df["final_gold_ok"] == 1) &
    (miti_df["delta_ok_score"] > 0.05) &          # ⭐ 防止假改对
    (miti_df["base_answer"] != miti_df["final_answer"])  # ⭐ 必须有变化
]

if len(improved) > 0:
    improved = improved.sample(
        n=min(NUM_CASES_PER_TYPE, len(improved)),
        random_state=SEED
    )
    improved["outcome"] = "Improved"


# ================================
# 2️⃣ Regressed（base对 → final错）
# ================================
regressed = miti_df[
    (miti_df["base_gold_ok"] == 1) &
    (miti_df["final_gold_ok"] == 0)
]

if len(regressed) > 0:
    regressed = regressed.sample(
        n=min(NUM_CASES_PER_TYPE, len(regressed)),
        random_state=SEED
    )
    regressed["outcome"] = "Regressed"


# ================================
# 3️⃣ Error（检测错误）
# ================================
error = error_df[
    error_df["pred_ok"] == 0
]

if len(error) > 0:
    error = error.sample(
        n=min(NUM_CASES_PER_TYPE, len(error)),
        random_state=SEED
    )

# 字段对齐
error = error.rename(columns={
    "answer": "base_answer"
})

error["final_answer"] = ""
error["outcome"] = "Error"


# ================================
# 合并
# ================================
frames = []

if len(improved) > 0:
    frames.append(improved)

if len(regressed) > 0:
    frames.append(regressed)

if len(error) > 0:
    frames.append(error)

cases = pd.concat(frames, ignore_index=True)


# ================================
# 去重（避免重复问题）
# ================================
cases = cases.drop_duplicates(subset=["question"])


# ================================
# 选择字段（论文用）
# ================================
cols = [
    "question",
    "base_answer",
    "final_answer",
    "outcome"
]

cases = cases[[c for c in cols if c in cases.columns]]


# ================================
# 保存
# ================================
cases.to_csv(OUTPUT_FILE, index=False)

print(f"✅ TQA case studies saved to {OUTPUT_FILE}")