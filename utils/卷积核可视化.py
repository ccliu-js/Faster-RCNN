import matplotlib.pyplot as plt
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

def visualize_conv_kernels(layer, num_cols=8, figsize=(12, 12)):
    """
    可视化任意卷积层的卷积核
    :param layer: 卷积层 (nn.Conv2d)
    :param num_cols: 每行显示多少个卷积核
    :param figsize: 图像大小
    """
    weights = layer.weight.data.cpu()
    out_channels, in_channels, kH, kW = weights.shape

    num_rows = (out_channels + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    for i, ax in enumerate(axes.flatten()):
        if i < out_channels:
            kernel = weights[i]
            kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())

            if in_channels == 3:  # RGB 卷积核
                kernel = kernel.permute(1, 2, 0)
                ax.imshow(kernel)
            else:  # 单通道
                ax.imshow(kernel[0], cmap="gray")

        ax.axis('off')

    plt.show()

# ✅ 使用新版写法加载预训练模型
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)

# 可视化 backbone 的第一层 conv1
visualize_conv_kernels(model.backbone.body.layer1)

# 还可以可视化更深层，比如 layer1[0].conv1
# visualize_conv_kernels(model.backbone.body.layer1[0].conv1)
visualize_conv_kernels(model.backbone.body.layer1[0].conv2)
