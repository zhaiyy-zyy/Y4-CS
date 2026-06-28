import pandas as pd

INPUT_FILE = "truthfulqa_mitigation_cases_all10.csv"

df = pd.read_csv(INPUT_FILE)

def is_abstain(ans):
    if not isinstance(ans, str):
        return True
    ans = ans.lower()
    return any(x in ans for x in [
        "i do not know",
        "don't know",
        "cannot determine",
        "not sure",
        "unknown"
    ])

# ===== 分类函数 =====
def classify(row):
    base_ok = bool(row["base_gold_ok"])
    final_ok = bool(row["final_gold_ok"])

    # ⭐ 用 pipeline 的结果，而不是字符串判断
    abstain = bool(row["final_NEI"])

    if abstain:
        return "abstain"
    elif (not base_ok) and final_ok:
        return "improved"
    elif base_ok and (not final_ok):
        return "regressed"
    elif base_ok and final_ok:
        return "preserved"
    else:
        return "still_wrong"
# ===== 应用 =====
df["case_type"] = df.apply(classify, axis=1)

# ===== 拆分保存 =====
df[df["case_type"] == "improved"].to_csv("cases_improved.csv", index=False)
df[df["case_type"] == "regressed"].to_csv("cases_regressed.csv", index=False)
df[df["case_type"] == "abstain"].to_csv("cases_abstain.csv", index=False)
df[df["case_type"] == "preserved"].to_csv("cases_preserved.csv", index=False)
df[df["case_type"] == "still_wrong"].to_csv("cases_still_wrong.csv", index=False)

print(df.columns)

# ===== 统计 =====
print("\n=== Summary ===")
print(df["case_type"].value_counts())