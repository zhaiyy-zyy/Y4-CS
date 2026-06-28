import matplotlib.pyplot as plt
import numpy as np

datasets = ["FEVER", "NQ", "TQA"]
thresholds = [0.225, 0.80, 0.465]
halluc = [0.295, 0.805, 0.886]

x = np.arange(len(datasets))
width = 0.35

fig, ax = plt.subplots() # 建议使用 subplots 模式，更方便操作坐标轴对象

# 将返回的容器赋值给变量
rects1 = ax.bar(x - width/2, thresholds, width, label="Threshold")
rects2 = ax.bar(x + width/2, halluc, width, label="Hallucination")

# 使用 bar_label 添加数值
# padding 表示数值距离柱体顶部的距离
ax.bar_label(rects1, padding=3, fmt='%.3f') 
ax.bar_label(rects2, padding=3, fmt='%.3f')

ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("Value")
ax.set_title("Dataset-wise Threshold and Hallucination Levels")

# 为了防止标签被顶部的边框挡住，可以手动调高 y 轴上限
ax.set_ylim(0, 1.1) 

ax.legend()
plt.savefig("fig_threshold_dataset.png", dpi=300, bbox_inches='tight')
plt.show()