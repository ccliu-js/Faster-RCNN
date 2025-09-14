import matplotlib.pyplot as plt
import torch
from torchvision import models
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# 获取 backbone 的第一层 conv 权重
conv1_weights = model.backbone.body.conv1.weight.data.cpu()

# conv1_weights.shape => [64, 3, 7, 7] (out_channels, in_channels, kH, kW)

fig, axes = plt.subplots(8, 8, figsize=(12, 12))
for i, ax in enumerate(axes.flatten()):
    # 将卷积核归一化到0~1
    kernel = conv1_weights[i]
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    # 转置通道为 HWC
    kernel = kernel.permute(1, 2, 0)
    ax.imshow(kernel)
    ax.axis('off')
plt.show()



