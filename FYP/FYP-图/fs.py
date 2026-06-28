import pandas as pd

# 读取数据
df = pd.read_csv("fever_mitigation_cases.csv")

# ===== 1️⃣ 清洗 =====
df = df.dropna(subset=["base_gold_ok", "final_gold_ok"])

df["base_gold_ok"] = df["base_gold_ok"].astype(bool)
df["final_gold_ok"] = df["final_gold_ok"].astype(bool)

# ===== ⭐ 2️⃣ 关键：只保留每个 question 一条 =====
df = df.sort_values("final_ok_score", ascending=False) \
       .drop_duplicates(subset=["question"])

# ===== 3️⃣ 分类 =====
def classify(row):
    if (not row["base_gold_ok"]) and row["final_gold_ok"]:
        return "fixed"
    elif row["base_gold_ok"] and (not row["final_gold_ok"]):
        return "regressed"
    elif (not row["base_gold_ok"]) and (not row["final_gold_ok"]):
        return "still_wrong"
    else:
        return "already_correct"

df["improvement_type"] = df.apply(classify, axis=1)

# ===== 4️⃣ sanity check =====
print("\nTotal questions:", len(df))
print("Correct:", df["final_gold_ok"].sum())
print("Hallucinated:", (~df["final_gold_ok"]).sum())

print("\nImprovement breakdown:")
print(df["improvement_type"].value_counts())

# ===== 5️⃣ 分别导出（每个 question 只出现一次）=====
for t in ["already_correct", "fixed", "regressed", "still_wrong"]:
    sub = df[df["improvement_type"] == t]
    sub.to_csv(f"{t}.csv", index=False)
    print(f"{t}: {len(sub)} saved")