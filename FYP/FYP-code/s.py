import pandas as pd

df = pd.read_csv("f_mitigation_cases9.csv")

# 1️⃣ 只保留 policy
df_policy = df[df["policy"] == "policy"].copy()

# 2️⃣ 每个问题只保留一个
df_q = df_policy.drop_duplicates(subset=["question"])

print("Total questions:", len(df_q))


# =========================
# 🧠 分类函数（FEVER版本）
# =========================
def classify(row, prefix="final"):
    
    gold_ok = row[f"{prefix}_gold_ok"]
    pred_label = row[f"{prefix}_pred_label"]
    answer = str(row[f"{prefix}_answer"]).lower()
    
    # =========================
    # ⭐ 1️⃣ 如果有合法label → 绝对不是abstain
    # =========================
    if pred_label in ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]:
        
        if gold_ok:
            if pred_label == "NOT ENOUGH INFO":
                return "Correct_NEI"
            else:
                return "Correct"
        else:
            return "Wrong"
    
    # =========================
    # 🚫 2️⃣ 没有label才判断abstain
    # =========================
    ABSTAIN_PATTERNS = [
        "i do not know",
        "don't know",
        "cannot answer",
        "not sure",
        "unknown"
    ]
    
    if any(p in answer for p in ABSTAIN_PATTERNS):
        return "Abstain"
    
    # fallback
    return "Abstain"


# =========================
# 分类
# =========================
df_q["final_category"] = df_q.apply(lambda x: classify(x, "final"), axis=1)


# =========================
# 统计
# =========================
dist = df_q["final_category"].value_counts(normalize=True)

print("\n===== FEVER RESULT =====")
print(dist)


# =========================
# Case study
# =========================
# =========================
# 4类分开（关键🔥）
# =========================

correct = df_q[df_q["final_category"] == "Correct"]

correct_nei = df_q[df_q["final_category"] == "Correct_NEI"]

wrong = df_q[df_q["final_category"] == "Wrong"]

abstain = df_q[df_q["final_category"] == "Abstain"]


# =========================
# 保存
# =========================

correct.to_csv("fever_correct.csv", index=False)
correct_nei.to_csv("fever_correct_nei.csv", index=False)
wrong.to_csv("fever_wrong.csv", index=False)
abstain.to_csv("fever_abstain.csv", index=False)

print("✅ 4-category case files saved")

dist = df_q["final_category"].value_counts(normalize=True)

print("\n===== FEVER RESULT (4 classes) =====")
print(dist)