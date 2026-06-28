import pandas as pd
import os

# ====== 1. 读取数据 ======
file_path = "f_mitigation_cases12.csv"
df = pd.read_csv(file_path)

# 创建输出文件夹
os.makedirs("classified_cases", exist_ok=True)

# ====== 2. 四类分类 ======

# 1) 改对了：原来错 -> 现在对
improved = df[
    (df["base_gold_ok"] == False) &
    (df["final_gold_ok"] == True)
]

# 2) 改坏了：原来对 -> 现在错
regressed = df[
    (df["base_gold_ok"] == True) &
    (df["final_gold_ok"] == False)
]

# 3) 仍然错：原来错 -> 现在还是错
still_wrong = df[
    (df["base_gold_ok"] == False) &
    (df["final_gold_ok"] == False)
]

# 4) 仍然对：原来对 -> 现在还是对
still_correct = df[
    (df["base_gold_ok"] == True) &
    (df["final_gold_ok"] == True)
]

# ====== 3. 保存 ======
improved.to_csv("classified_cases/improved.csv", index=False)
regressed.to_csv("classified_cases/regressed.csv", index=False)
still_wrong.to_csv("classified_cases/still_wrong.csv", index=False)
still_correct.to_csv("classified_cases/still_correct.csv", index=False)

# ====== 4. 打印数量检查 ======
print("improved:", len(improved))
print("regressed:", len(regressed))
print("still_wrong:", len(still_wrong))
print("still_correct:", len(still_correct))
print("total:", len(df))
print("sum:", len(improved) + len(regressed) + len(still_wrong) + len(still_correct))

print("🎉 四类分类完成！")