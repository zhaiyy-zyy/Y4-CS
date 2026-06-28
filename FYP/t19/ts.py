import pandas as pd

# ===== 1. 读取数据 =====
file_path = "truthfulqa_mitigation_cases_all23.csv"
df = pd.read_csv(file_path)

print("Total rows:", len(df))

# ===== 2. 筛选 regressed =====
# base 正确 & final 错误
regressed = df[(df["base_gold_ok"]) & (~df["final_gold_ok"])]

print("Regressed cases:", len(regressed))

# ===== 3. 查看前几个 =====
print("\n=== First 5 regressed cases ===")
print(regressed[["question", "base_answer", "final_answer"]].head(5))

# ===== 4. 随机抽样（写论文用）=====
print("\n=== Random 5 samples ===")
samples = regressed.sample(min(5, len(regressed)))
print(samples[["question", "base_answer", "final_answer"]])

# ===== 5. 导出 CSV =====
regressed.to_csv("tqa_regressed_cases.csv", index=False)
print("\nSaved to: tqa_regressed_cases.csv")

# ===== 6.（可选）统计分布 =====
correct_unchanged = df[(df["base_gold_ok"]) & (df["final_gold_ok"])]
incorrect_unchanged = df[(~df["base_gold_ok"]) & (~df["final_gold_ok"])]
abstained = df[df["final_answer"].str.contains("I do not know", na=False)]

total = len(df)

print("\n=== Outcome distribution ===")
print("Regressed:", len(regressed)/total)
print("Correct unchanged:", len(correct_unchanged)/total)
print("Incorrect unchanged:", len(incorrect_unchanged)/total)
print("Abstained:", len(abstained)/total)