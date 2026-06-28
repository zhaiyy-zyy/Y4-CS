import matplotlib.pyplot as plt
import numpy as np

# 每个密码一个柱子
passwords = [
    "abc123", "def456",
    "abcd1234", "wxyz5678",
    "abcde12345", "fghij67890",
    "abcdef123456", "ghijkl789012"
]

# cracking time（秒）
times = [
    0.835, 58.615,
    1.082, 11000,
    13.705, 10800,
    179, 9760
]

# 状态
status = [
    "Completed", "Completed",
    "Completed", "Aborted (48.43%)",
    "Completed", "Aborted (46.17%)",
    "Completed", "Aborted (43.07%)"
]

colors = [
    "tab:blue", "tab:orange",
    "tab:blue", "tab:orange",
    "tab:blue", "tab:orange",
    "tab:blue", "tab:orange"
]

x = np.arange(len(passwords))

plt.figure(figsize=(13, 10))
bars = plt.bar(x, times, color=colors)

# log scale
plt.yscale("log")

plt.title("Effect of Password Length on Cracking Time")
plt.xlabel("Password")
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
group_labels = ["6-character", "8-character", "10-character", "12-character"]

for xc, label in zip(group_centers, group_labels):
    plt.text(
    xc, -0.08, label,
    transform=plt.gca().get_xaxis_transform()
)

plt.subplots_adjust(bottom=0.25)  # 给底部分组标题留空间
plt.tight_layout()
plt.savefig("length.png", dpi=300, bbox_inches="tight")
plt.show()