import numpy as np
import pandas as pd
import cv2
from skimage import measure
import matplotlib.pyplot as plt

# 解码 RLE 编码为二值掩码
def rle_decode(rle, shape):
    """Decode Run-Length Encoding (RLE) to binary mask."""
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    rle = list(map(int, rle.split()))
    for i in range(0, len(rle), 2):
        img[rle[i]:rle[i] + rle[i + 1]] = 1
    return img.reshape(shape).T

# 从掩码中提取边界框
def get_bounding_boxes(mask):
    """Get bounding boxes from binary mask."""
    # Label connected components (regions of interest)
    labels = measure.label(mask)
    regions = measure.regionprops(labels)
    boxes = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        boxes.append([minc, minr, maxc, maxr])  # [x_min, y_min, x_max, y_max]
    return boxes

# 读取 train.csv 文件并解析
train_df = pd.read_csv('train.csv')

# 获取一张图像的数据（例如，第一行）
sample = train_df.iloc[0]
image_id = sample['image_id']
encoded_pixels = sample['EncodedPixels']
shape = (256, 1600)  # 假设图像大小为 256x1600

# 解码 RLE 获得掩码
mask = rle_decode(encoded_pixels, shape)

# 提取边界框
bounding_boxes = get_bounding_boxes(mask)
print(f"Bounding boxes for {image_id}: {bounding_boxes}")

# 可视化掩码和边界框
image = np.zeros(shape, dtype=np.uint8)  # 创建一个空白图像来显示掩码
image[mask == 1] = 255  # 将掩码区域设置为白色

# 创建一个图形来显示掩码和边界框
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.imshow(image, cmap='gray')

# 绘制边界框
for box in bounding_boxes:
    x_min, y_min, x_max, y_max = box
    ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               linewidth=2, edgecolor='r', facecolor='none'))

plt.show()
