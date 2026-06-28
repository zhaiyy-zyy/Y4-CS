import matplotlib.pyplot as plt
import numpy as np

# 每个密码一个柱子
passwords = [
    "123456", "password", "admin", "qwerty",
    "T9#KLp2@vX1q", "Xk7!pL9@qR2"
]


# cracking time（秒）
times = [
    0.951, 0.775, 6.059, 0.837,   # weak
    13403, 13082       # strong（>2h）
]

# 状态
status = [
    "Completed", "Completed", "Completed", "Completed",
    "Aborted (65.32%)", "Aborted (77.7%)"
]

colors = [
    "tab:blue", "tab:blue", "tab:blue", "tab:blue",   # weak
    "tab:orange", "tab:orange"                # strong
]


x = np.arange(len(passwords))

plt.figure(figsize=(13, 10))
bars = plt.bar(x, times, color=colors)

# log scale
plt.yscale("log")

plt.title("Effect of Password Strength (Weak vs Strong) on Cracking Time")
plt.xlabel("Password", labelpad=25)
plt.ylabel("Cracking Time (seconds, log scale)")

# 只保留密码名
plt.xticks(x, passwords, rotation=25)

# 标注柱子
for i in range(len(passwords)):
    if "Aborted" in status[i]:
        plt.text(
            x[i], times[i] * 1.1,
            "Aborted\n>2h",
            ha="center", fontsize=9
        )
    else:
        plt.text(
            x[i], times[i] * 1.2,
            f"Completed\n{times[i]:.2f}s",
            ha="center", fontsize=9
        )

# 在底部加分组分隔线
plt.axvline(3.5, color="gray", linestyle="--", alpha=0.6)
# 在图底部用 axis transform 放组标题
group_centers = [1.5, 5]
group_labels = ["Weak Passwords", "Strong Passwords"]

for xc, label in zip(group_centers, group_labels):
    plt.text(
        xc, -0.09, label,
        transform=plt.gca().get_xaxis_transform(),
        ha="center",
        fontsize=11
    )

plt.subplots_adjust(bottom=0.35)  # 给底部分组标题留空间
plt.tight_layout()
plt.savefig("weak_vs_strong.png", dpi=300, bbox_inches="tight")
plt.show()