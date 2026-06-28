import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

# =========================
# 1️⃣ 数据（与你代码输出一致）
# =========================
data = {
    'Dataset': ['FEVER']*3 + ['NQ']*3 + ['TQA']*4,

    'Policy': [
        # FEVER
        'Mild', 'Strict-K', 'Final',

        # NQ
        'Extract-1', 'Extract-K', 'Cond-Ex',

        # TQA
        'Rewrite-1', 'Rewrite-K', 'Cond-Truth', 'Cond-Util'
    ],

    'Delta_OK': [
        # FEVER
        0.0198, 0.1061, 0.1075,

        # NQ
        0.0013, 0.0013, 0.0695,

        # TQA
        -0.0081, -0.0572, 0.0008, 0.0004
    ],

    'Hallu_Reduc': [
        # FEVER
        0.0383, 0.3016, 0.3016,

        # NQ
        0.0000, 0.0000, 0.0433,

        # TQA
        0.0050, -0.0200, 0.0067, 0.0067
    ]
}

df = pd.DataFrame(data)
datasets = ['FEVER', 'NQ', 'TQA']

# =========================
# 2️⃣ Dataset颜色（关键改动）
# =========================
dataset_colors = {
    'FEVER': '#5DADE2',   # 蓝
    'NQ': '#F5B041',      # 橙
    'TQA': '#58D68D'      # 绿
}

# =========================
# 3️⃣ 画图设置
# =========================
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))  # ⭐ 大图

# =========================
# 4️⃣ 绘图函数
# =========================
def plot_data(ax, col_name, title):

    current_x = 0
    tick_pos = []

    for d in datasets:
        subset = df[df['Dataset'] == d]
        n = len(subset)

        pos = np.arange(current_x, current_x + n)
        tick_pos.append(pos.mean())

        values = subset[col_name].values

        # ⭐ dataset统一颜色
        bars = ax.bar(
            pos,
            values,
            color=dataset_colors[d],
            edgecolor='black',
            alpha=0.85
        )

        # ⭐ 负值加斜线（更专业）
        for i, bar in enumerate(bars):
            if values[i] < 0:
                bar.set_hatch('//')

        # 数值标签
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3 if h >= 0 else -12),
                        textcoords="offset points",
                        ha='center',
                        fontsize=9,
                        fontweight='bold')

        # policy标签（避免重叠）
        for i, p in enumerate(subset['Policy']):
            ax.text(pos[i], -0.07,
                    p,
                    ha='right',
                    va='top',
                    rotation=45,
                    fontsize=10)

        # ⭐ 拉开dataset间距
        current_x += n + 2

    # dataset标签
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(datasets, fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', pad=60)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axhline(0, color='black', linewidth=1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

# =========================
# 5️⃣ 左图 ΔOK
# =========================
plot_data(ax1, 'Delta_OK', "Mitigation Gain (ΔOK)")
ax1.set_ylabel("Score Increase", fontsize=13)
ax1.set_ylim(-0.08, 0.16)

# =========================
# 6️⃣ 右图 Hallucination Reduction
# =========================
plot_data(ax2, 'Hallu_Reduc', "Hallucination Reduction")
ax2.set_ylabel("Reduction Rate", fontsize=13)
ax2.set_ylim(-0.08, 0.36)

# =========================
# 7️⃣ Legend（dataset）
# =========================
legend_elements = [
    Patch(facecolor='#5DADE2', edgecolor='black', label='FEVER'),
    Patch(facecolor='#F5B041', edgecolor='black', label='NQ'),
    Patch(facecolor='#58D68D', edgecolor='black', label='TQA'),
    Patch(facecolor='white', edgecolor='black', hatch='//', label='Negative Effect')
]

fig.legend(handles=legend_elements,
           loc='upper center',
           ncol=4,
           fontsize=12)

# =========================
# 8️⃣ 输出
# =========================
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("mitigation_final_professional.png", dpi=300, bbox_inches='tight')
plt.show()