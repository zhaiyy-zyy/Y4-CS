import pandas as pd

def load_dataset_summary(path, dataset_name):
    df = pd.read_csv(path)

    # ⭐ 清理列名（防止空格问题）
    df.columns = df.columns.str.strip()

    # ⭐ 统一 mode 名字
    mode_map = {
        "full": "Full",
        "no_judge": "No Judge",
        "no_mnli": "No MNLI",
        "no_selfcheck": "No Self-check",
        "judge_only": "Judge Only"
    }

    df['Mode'] = df['mode'].map(mode_map)
    df['Dataset'] = dataset_name

    # ⭐ 自动识别列名（适配三种数据集）
    if 'F1_mean' in df.columns:
        f1_col = 'F1_mean'
    elif 'F1' in df.columns:
        f1_col = 'F1'
    else:
        raise ValueError(f"{dataset_name}: F1 column not found")

    if 'A_std' in df.columns:
        std_col = 'A_std'
    elif 'A-std' in df.columns:
        std_col = 'A-std'
    else:
        raise ValueError(f"{dataset_name}: A_std column not found")

    # ⭐ 返回统一格式（关键）
    return df[['Dataset', 'Mode', f1_col, std_col]].rename(
        columns={
            f1_col: 'F1_mean',
            std_col: 'A_std'
        }
    )


# ===== 加载三个数据集 =====
fever_df = load_dataset_summary("fever_paper_overall_summary.csv", "FEVER")
nq_df = load_dataset_summary("nq_paper_overall_summary.csv", "NQ")
tqa_df = load_dataset_summary("truthfulqa_paper_overall_summary.csv", "TQA")

# ===== 合并 =====
df = pd.concat([fever_df, nq_df, tqa_df], ignore_index=True)

# ⭐ 检查结果（建议保留）
print(df.head())