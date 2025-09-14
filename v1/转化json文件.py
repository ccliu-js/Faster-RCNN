import os
import json

# 配置
images_folder = "dataset/images"
labels_folder = "dataset/labels"
output_folder = "dataset/processed_labels"
os.makedirs(output_folder, exist_ok=True)

# 类别映射，可根据你的实际类别修改
label_to_index = {
  "气孔": 1,
  "沟槽": 2,
  "破碎": 3,
  "蜂窝": 4,
  "麻面": 5
}


# 遍历所有 JSON 文件
for json_file in os.listdir(labels_folder):
    if not json_file.endswith(".json"):
        continue

    json_path = os.path.join(labels_folder, json_file)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes = []
    labels = []

    for shape in data.get("shapes", []):
        if shape["shape_type"] != "rectangle":
            continue  # 只处理矩形框

        pts = shape["points"]
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        boxes.append([x1, y1, x2, y2])

        label_name = shape["label"]
        labels.append(label_to_index.get(label_name, 0))  # 未知类别用 0

    processed_data = {
        "imagePath": data["imagePath"],
        "height": data.get("imageHeight"),
        "width": data.get("imageWidth"),
        "boxes": boxes,
        "labels": labels
    }

    # 输出新 JSON
    output_path = os.path.join(output_folder, json_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

print("转换完成，处理后的 JSON 已保存到 processed_labels 文件夹")

input("请按 enter 终止程序")
