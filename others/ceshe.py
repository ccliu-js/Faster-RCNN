from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

# 加载预训练 Faster R-CNN
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)

# 取出 layer1[0] 的 bottleneck
bottleneck = model.backbone.body.layer1[0]

# 拿到里面的 3x3 卷积层
conv3x3 = bottleneck.conv2

# 可视化
visualize_conv_kernels(conv3x3, num_cols=8, figsize=(12, 12))