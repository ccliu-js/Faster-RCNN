import torch
from PIL import Image
import os
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from _model import FasterRCNNModel  # 假设你的模型类存储在 _model.py 文件中

num_class = 6     # 别忘了背景也算一类
confidence = 0.6
weight_url = 'weight/best_model.pth'
testData_url = 'testData'

# 1. 加载模型
model = FasterRCNNModel(num_classes=num_class, pretrained=False)  # 预训练模型关闭，因为我们已加载自己的权重
model.load_state_dict(torch.load(weight_url, map_location=model.device))  # 加载训练好的权重
model.to(model.device)  # 移动到合适的设备
model.eval()  # 切换到推理模式

# 2. 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为 Tensor
])

# 获取文件夹中所有的图片文件
image_files = [f for f in os.listdir(testData_url) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 3. 创建保存结果的文件夹
os.makedirs('inferData', exist_ok=True)  # 创建推理结果保存目录（改为一致的目录）

# 遍历每张图片文件
for image_file in image_files:
    # 加载图像
    image_path = os.path.join(testData_url, image_file)
    image = Image.open(image_path)

    # 将图像转换为 Tensor 格式
    image_tensor = transform(image).to(model.device)  # 不再增加批次维度

    # 推理
    with torch.no_grad():  # 禁用梯度计算
        predictions = model([image_tensor])  # 传入的是包含一张图片的列表

    # 获取预测结果
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    # 可视化推理结果
    fig, ax = plt.subplots(1)

    # 将图像移到 CPU，再转换为 NumPy 数组
    image = image.convert("RGB")  # 确保图像是 RGB 格式
    image_np = np.array(image)  # 转换为 NumPy 数组

    ax.imshow(image_np)  # 显示原始图像

    # 绘制边界框，调整线宽
    for box, label, score in zip(boxes, labels, scores):
        if score > confidence:  # 只显示分数大于 0.5 的预测
            rect = patches.Rectangle(
                (box[0].item(), box[1].item()),
                box[2].item() - box[0].item(),
                box[3].item() - box[1].item(),
                linewidth=0.3,  # 调整线宽
                edgecolor='r',  # 红色边框
                facecolor='none'  # 不填充颜色
            )
            ax.add_patch(rect)
            ax.text(
                box[0].item(), box[1].item(),
                f'{label.item()} {score.item():.2f}',
                color='red', fontsize=5
            )

    # 保存结果
    save_path = os.path.join('inferData', f'{image_file}_result.jpg')  # 确保保存路径一致
    plt.savefig(save_path)  # 保存图像
    plt.close()  # 关闭当前绘图，释放内存
