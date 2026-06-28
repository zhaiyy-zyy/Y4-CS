import pandas as pd

# 读取文件
df = pd.read_csv("truthfulqa_mitigation_cases_all14.csv")

# ========= ⚠️ 关键：先统一成字符串再转bool =========
def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ["true", "1", "yes"]
    return bool(x)

df["base_gold_ok"] = df["base_gold_ok"].apply(to_bool)
df["final_gold_ok"] = df["final_gold_ok"].apply(to_bool)
df["final_NEI"] = df["final_NEI"].apply(to_bool)

# ========= 分类 =========
df["improve"] = (~df["base_gold_ok"]) & (df["final_gold_ok"])
df["regress"] = (df["base_gold_ok"]) & (~df["final_gold_ok"])
df["abstain"] = df["final_NEI"]

df["still_wrong"] = (~df["base_gold_ok"]) & (~df["final_gold_ok"]) & (~df["final_NEI"])
df["still_correct"] = (df["base_gold_ok"]) & (df["final_gold_ok"])

# ========= 统计 =========
summary = {
    "improve": int(df["improve"].sum()),
    "regress": int(df["regress"].sum()),
    "abstain": int(df["abstain"].sum()),
    "still_wrong": int(df["still_wrong"].sum()),
    "still_correct": int(df["still_correct"].sum()),
}

print(summary)

# ========= 分别保存 =========
df[df["improve"]].to_csv("cases_improve.csv", index=False)
df[df["regress"]].to_csv("cases_regress.csv", index=False)
df[df["abstain"]].to_csv("cases_abstain.csv", index=False)
df[df["still_wrong"]].to_csv("cases_still_wrong.csv", index=False)
df[df["still_correct"]].to_csv("cases_still_correct.csv", index=False)

print("✅ 已分别保存 5 个分类文件")