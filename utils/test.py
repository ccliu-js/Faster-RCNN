import torch
import torch.nn as nn
import torchvision.models.detection as models
import matplotlib.pyplot as plt

# 加载预训练 Faster R-CNN
model = models.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def visualize_all_conv_kernels(model, max_kernels_per_layer=8):
    """
    可视化 Faster R-CNN backbone 中所有卷积层的卷积核
    """
    # 获取 backbone 的 body (ResNet50)
    conv_modules = [(name, m) for name, m in model.backbone.body.named_modules() if isinstance(m, nn.Conv2d)]
    
    for idx, (name, conv) in enumerate(conv_modules):
        weights = conv.weight.data.cpu()
        num_kernels = min(weights.shape[0], max_kernels_per_layer)
        
        # 设置画布
        fig, axes = plt.subplots(1, num_kernels, figsize=(num_kernels*2, 2))
        if num_kernels == 1:
            axes = [axes]
        
        for i in range(num_kernels):
            kernel = weights[i]
            
            # 如果输入通道是3，用彩色显示；否则取第一个通道用灰度显示
            if kernel.shape[0] == 3:
                kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())  # 归一化到0~1
                kernel = kernel.permute(1, 2, 0)  # HWC
                axes[i].imshow(kernel)
            else:
                kernel = kernel[0]
                kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
                axes[i].imshow(kernel, cmap='gray')
            
            axes[i].axis('off')
        
        plt.suptitle(f"Conv Layer: {name}")
        plt.show()

# 调用函数
visualize_all_conv_kernels(model, max_kernels_per_layer=8)
