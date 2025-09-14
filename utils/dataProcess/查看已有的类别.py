import os
import json

labels_folder = "dataset/labels"  # 你的 LabelMe JSON 文件夹
all_labels = set()

# 遍历 JSON 文件
for json_file in os.listdir(labels_folder):
    if not json_file.endswith(".json"):
        continue

    with open(os.path.join(labels_folder, json_file), "r", encoding="utf-8") as f:
        data = json.load(f)

    for shape in data.get("shapes", []):
        label_name = shape.get("label")
        if label_name:
            all_labels.add(label_name)

# 输出所有类别
print("数据集中所有类别：")
for label in sorted(all_labels):
    print(label)
