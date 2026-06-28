import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 数据
data = {

    'Dataset': ['FEVER']*5 + ['NQ']*5 + ['TQA']*5,

    'Mode': ['Full', 'No Judge', 'No MNLI', 'No Self-check', 'Judge Only']*3,

    'F1_Mean': [

        # ===== FEVER（✔ 已核对）=====
        0.9606, 0.7587, 0.9606, 0.9606, 0.9606,

        # ===== NQ（✔ 已核对）=====
        0.9727, 0.9686, 0.9880, 0.9727, 0.9467,

        # ===== TQA（✔ 已核对）=====
        0.9153, 0.9196, 0.9196, 0.9196, 0.9196

    ],

    'A_std': [

        # ===== FEVER（✔ 已核对）=====
        0.1685, 0.0382, 0.3064, 0.2068, 0.4728,

        # ===== NQ（✔ 已核对）=====
        0.3277, 0.3369, 0.3079, 0.3240, 0.3409,

        # ===== TQA（✔ 已核对）=====
        0.1093, 0.1213, 0.0623, 0.1097, 0.0632

    ]
}

df = pd.DataFrame(data)

datasets = ['FEVER', 'NQ', 'TQA']
modes = ['Full', 'No Judge', 'No MNLI', 'No Self-check', 'Judge Only']

x = np.arange(len(datasets))
width = 0.15

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ===== F1 =====
for i, mode in enumerate(modes):
    values = [df[(df['Dataset']==d) & (df['Mode']==mode)]['F1_Mean'].values[0] for d in datasets]
    bars = ax1.bar(x + i*width - 0.3, values, width, label=mode)

    # ⭐ 加这一行
    ax1.bar_label(bars, fmt='%.2f', fontsize=8)

ax1.set_title("F1 Performance")
ax1.set_ylabel("F1 Score")
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=11)
ax1.set_ylim(0.7, 1.0)  # ⭐更清晰
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# ===== A-std =====
for i, mode in enumerate(modes):
    values = [df[(df['Dataset']==d) & (df['Mode']==mode)]['A_std'].values[0] for d in datasets]
    bars = ax2.bar(x + i*width - 0.3, values, width, label=mode)

    # ⭐ 加这一行
    ax2.bar_label(bars, fmt='%.2f', fontsize=8)

ax2.set_title("Stability (A-std)")
ax2.set_ylabel("Lower is Better")
ax2.set_xticks(x)
ax2.set_xticklabels(datasets, fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# ⭐ legend放中间（更适合PPT）
fig.legend(loc='upper center', ncol=5, fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig('clean_ablation.png', dpi=300)
plt.show()