import pandas as pd
import os

# ===== 读取数据 =====
df = pd.read_csv("fever_mitigation_cases.csv")

# ===== 提取 policy =====
policy_df = df[df["policy"] == "policy"]

print("Policy rows:", len(policy_df))

# ===== 创建输出文件夹 =====
output_dir = "policy_split"
os.makedirs(output_dir, exist_ok=True)

# ===== 用 improvement_type 分类 =====
types = policy_df["improvement_type"].unique()

for t in types:
    subset = policy_df[policy_df["improvement_type"] == t]
    
    filename = f"{output_dir}/policy_{t.replace(' ', '_')}.csv"
    subset.to_csv(filename, index=False)
    
    print(f"Saved {t}: {len(subset)} rows")

print("Done.")