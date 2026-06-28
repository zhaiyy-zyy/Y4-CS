import matplotlib.pyplot as plt
import numpy as np

# 每个密码一个柱子
passwords = [
    "12345678", "87654321",
    "abcdefgh", "ijklmnop",
    "abcd1234", "efgh5678",
    "Abc1@Def", "XyZ9#LmN"
]

# cracking time（秒）
times = [
    1.303, 0.966,      # numeric
    1.646, 1056,       # lowercase (~17min36s ≈ 1056s)
    0.980, 3321,       # alphanumeric (~55min ≈ 3321s)
    10931, 8356        # complex（>2h）
]

# 状态
status = [
    "Completed", "Completed",
    "Completed", "Completed",
    "Completed", "Completed",
    "Aborted (57.67%)", "Aborted (46.12%)"
]

colors = [
    "tab:blue", "tab:blue",        # numeric
    "tab:green", "tab:green",      # lowercase
    "tab:orange", "tab:orange",    # alphanumeric
    "tab:red", "tab:red"           # complex
]

x = np.arange(len(passwords))

plt.figure(figsize=(13, 10))
bars = plt.bar(x, times, color=colors)

# log scale
plt.yscale("log")

plt.title("Effect of Password Complexity on Cracking Time")
plt.xlabel("Password", labelpad=25)
plt.ylabel("Cracking Time (seconds, log scale)")

# 只保留密码名
plt.xticks(x, passwords, rotation=25)

# 标注柱子
for i in range(len(passwords)):
    if "Aborted" in status[i]:
        plt.text(
            x[i], times[i] * 1.1,
            f"{status[i]}\n>2h",
            ha="center", va="bottom", fontsize=9
        )
    else:
        plt.text(
            x[i], times[i] * 1.15,
            f"{status[i]}\n{times[i]:.1f}s",
            ha='center', va='bottom', fontsize=9
        )

# 在底部加分组分隔线
for pos in [1.5, 3.5, 5.5]:
    plt.axvline(pos, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

# 在图底部用 axis transform 放组标题
group_centers = [0.5, 2.5, 4.5, 6.5]
group_labels = ["Numeric", "Lowercase", "Alphanumeric", "High Complexity"]

for xc, label in zip(group_centers, group_labels):
    plt.text(
    xc, -0.08, label,
    transform=plt.gca().get_xaxis_transform()
)

plt.subplots_adjust(bottom=0.35)  # 给底部分组标题留空间
plt.tight_layout()
plt.savefig("complexity.png", dpi=300, bbox_inches="tight")
plt.show()