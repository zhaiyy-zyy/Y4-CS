import pandas as pd
import os

# 输入文件
INPUT_FILE = "f_mitigation_cases10.csv"

# 输出目录
OUT_DIR = "mitigation_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# 读取数据
df = pd.read_csv(INPUT_FILE)

print("Total cases:", len(df))
print(df["improvement_type"].value_counts())

# ========= 分类 =========
fixed_df = df[df["improvement_type"] == "fixed"]
regressed_df = df[df["improvement_type"] == "regressed"]
still_wrong_df = df[df["improvement_type"] == "still_wrong"]
already_correct_df = df[df["improvement_type"] == "already_correct"]

# ========= 保存 =========
fixed_df.to_csv(os.path.join(OUT_DIR, "fixed_cases.csv"), index=False)
regressed_df.to_csv(os.path.join(OUT_DIR, "regressed_cases.csv"), index=False)
still_wrong_df.to_csv(os.path.join(OUT_DIR, "still_wrong_cases.csv"), index=False)
already_correct_df.to_csv(os.path.join(OUT_DIR, "already_correct_cases.csv"), index=False)

print("\nSaved files:")
print(" - fixed_cases.csv")
print(" - regressed_cases.csv")
print(" - still_wrong_cases.csv")
print(" - already_correct_cases.csv")