import torch
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

# -------------------------------
# 类别映射
# -------------------------------
label_to_index = {
    "气孔": 1,
    "沟槽": 2,
    "破碎": 3,
    "蜂窝": 4,
    "麻面": 5
}
num_classes = len(label_to_index) + 1  # +1 背景类

# -------------------------------
# 加载预训练 Faster R-CNN
# -------------------------------
weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)

# -------------------------------
# 自定义 Anchor Generator
# -------------------------------
anchor_generator = AnchorGenerator(
    sizes=((16,), (32,), (64,), (128,), (256,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)
model.rpn.anchor_generator = anchor_generator

# -------------------------------
# 替换预测头
# -------------------------------
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# -------------------------------
# 测试前向
# -------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

dummy_image = torch.rand(3, 512, 512).to(device)  # 单张图
model.eval()
with torch.no_grad():
    output = model([dummy_image])  # 输入必须是列表
print("✅ 测试成功, 输出 keys:", output[0].keys())
