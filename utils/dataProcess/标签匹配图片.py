import os

# 文件夹路径
images_folder = "dataset/images"
labels_folder = "dataset/labels"

# 获取文件名（不带后缀）
image_files = set(os.path.splitext(f)[0] for f in os.listdir(images_folder) if f.endswith(".jpg"))
label_files = set(os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.endswith(".json"))

# 找到同时存在的文件
common_files = image_files & label_files

# 删除不匹配的图片
for f in image_files - common_files:
    os.remove(os.path.join(images_folder, f + ".jpg"))

# 删除不匹配的标签
for f in label_files - common_files:
    os.remove(os.path.join(labels_folder, f + ".json"))

print(f"完成！共保留 {len(common_files)} 对文件。")

print(f"完成！共保留 {len(common_files)} 对文件。")

input("回车退出程序......")