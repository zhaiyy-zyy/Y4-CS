# ==============================
# 7. 验证输出是否正确
# ==============================
print("\n🧪 开始验证输出结果...")

for col in COLUMNS_TO_SPLIT:
    col_dir = os.path.join(OUTPUT_DIR, col)

    if not os.path.exists(col_dir):
        print(f"❌ 文件夹不存在: {col_dir}")
        continue

    print(f"\n🔍 检查列: {col}")

    total_rows = 0
    unique_ids = set()

    for file in os.listdir(col_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(col_dir, file)
            temp_df = pd.read_csv(file_path)

            rows = len(temp_df)
            total_rows += rows

            # 用 q_idx + run 做唯一标识（更安全）
            if "q_idx" in temp_df.columns and "run" in temp_df.columns:
                ids = set(zip(temp_df["run"], temp_df["q_idx"]))
                unique_ids.update(ids)

            print(f"  📄 {file}: {rows} rows")

    original_rows = len(df)

    print(f"\n  👉 原始数据: {original_rows}")
    print(f"  👉 分类后总和: {total_rows}")
    print(f"  👉 唯一数据数: {len(unique_ids)}")

    # 判断是否正确
    if total_rows == original_rows:
        print("  ✅ 行数正确（没有丢失）")
    else:
        print("  ❌ 行数不一致（有问题！）")

    if len(unique_ids) == original_rows:
        print("  ✅ 无重复数据")
    else:
        print("  ❌ 存在重复数据")