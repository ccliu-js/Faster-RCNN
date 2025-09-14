import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FasterRCNNModel(torch.nn.Module):
    def __init__(self, num_classes=6, pretrained=True, device=None):
        super(FasterRCNNModel, self).__init__()
        
        # backbone + FPN
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT" if pretrained else None
        )

        # 替换分类头
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # 设备
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, images, targets=None):
        images = [img.to(self.device) for img in images]
        if targets is not None:
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            return self.model(images, targets)  # 训练模式：返回 loss
        else:
            return self.model(images)  # 推理模式：返回预测结果


