import json

# 读取两个JSON文件
with open("docs/2wikimultihopqa_selection_ann_threshold.json", "r", encoding="utf-8") as f_sel:
    selection_data = json.load(f_sel)

with open("docs/2wikimultihopqa_backup.json", "r", encoding="utf-8") as f_backup:
    backup_data = json.load(f_backup)

# 获取各自的后200条（如果长度不足200，就取全部）
selection_last_200 = selection_data#[-200:]
backup_last_200 = backup_data[-200:]

# 合并数据：selection的在前，backup的在后
merged_data = selection_last_200 + backup_last_200

# 保存到新文件中
with open("docs/2wikimultihopqa_after_sele.json", "w", encoding="utf-8") as f_out:
    json.dump(merged_data, f_out, ensure_ascii=False, indent=2)

print(f"成功写入 {len(merged_data)} 条数据到 docs/2wikimultihopqa_after_sele.json")
