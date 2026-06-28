import pandas as pd

df = pd.read_csv("nq_mitigation_cases_final_all7.csv")

# policy + 去重
df_q = df[df["policy"] == "policy"].drop_duplicates(subset=["question"])

print("Total unique questions:", len(df_q))

# =========================
# 1️⃣ 分是否回答（最重要）
# =========================
abstain = df_q[df_q["final_NEI"] == True]
answered = df_q[df_q["final_NEI"] == False]

# =========================
# 2️⃣ answered 内再分类
# =========================
fixed = answered[
    (answered["base_gold_ok"] == False) &
    (answered["final_gold_ok"] == True)
]

regress = answered[
    (answered["base_gold_ok"] == True) &
    (answered["final_gold_ok"] == False)
]

still_wrong = answered[
    (answered["base_gold_ok"] == False) &
    (answered["final_gold_ok"] == False)
]

unchanged_correct = answered[
    (answered["base_gold_ok"] == True) &
    (answered["final_gold_ok"] == True)
]

# =========================
# 3️⃣ 打印
# =========================
print("\n===== STATS =====")
print("ABSTAIN:", len(abstain))
print("ANSWERED:", len(answered))
print("FIXED:", len(fixed))
print("REGRESS:", len(regress))
print("STILL WRONG:", len(still_wrong))
print("UNCHANGED CORRECT:", len(unchanged_correct))

# =========================
# 5️⃣ 检查是否覆盖全部
# =========================

total = len(df_q)
sum_all = len(fixed) + len(regress) + len(abstain) + len(still_wrong) + len(unchanged_correct)

print("\n===== CHECK =====")
print("Total:", total)
print("Sum:", sum_all)

# =========================
# 6️⃣ 导出case study
# =========================

fixed.to_csv("nq_fixed.csv", index=False)
regress.to_csv("nq_regress.csv", index=False)
abstain.to_csv("nq_abstain.csv", index=False)
still_wrong.to_csv("nq_still_wrong.csv", index=False)
unchanged_correct.to_csv("nq_unchanged_correct.csv", index=False)

print("✅ Case files saved")

# =========================
# 7️⃣ 过滤掉 abstain（关键！）
# =========================

df_answered = df_q[df_q["final_NEI"] == False]

print("\nAnswered subset:", len(df_answered))

if len(df_answered) == 0:
    print("\n⚠️ No answered samples — model abstained on ALL questions.")
    print("⚠️ Cannot compute hallucination rate excluding abstain.")

    base_halluc_rate = None
    final_halluc_rate = None
    improvement = None

else:
    # base hallucination
    base_halluc = df_answered[df_answered["base_gold_ok"] == False]
    base_halluc_rate = len(base_halluc) / len(df_answered)

    # final hallucination
    final_halluc = df_answered[df_answered["final_gold_ok"] == False]
    final_halluc_rate = len(final_halluc) / len(df_answered)

    # improvement
    improvement = base_halluc_rate - final_halluc_rate

    print("\n===== HALLUCINATION (excluding abstain) =====")
    print("Base halluc rate:", base_halluc_rate)
    print("Final halluc rate:", final_halluc_rate)
    print("Improvement:", improvement)