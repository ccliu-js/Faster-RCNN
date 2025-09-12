import torch
from torchvision import models
from torchvision.transforms import functional as F
from PIL import Image

# 加载预训练的 Faster R-CNN 模型（ResNet50 + FPN）
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # 切换到评估模式

# 读取并处理图像
image = Image.open("R.jpg")
image_tensor = F.to_tensor(image).unsqueeze(0)  # 转换为 Tensor，并增加 batch 维度

# 使用模型进行推理
with torch.no_grad():  # 推理时不计算梯度
    predictions = model(image_tensor)



# 输出检测结果
print(predictions)
