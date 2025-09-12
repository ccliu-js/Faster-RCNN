import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FastRCNNPredictor

class MyFasterRCNN(torch.nn.Module):
    def __init__(self, num_classes=6, pretrained=True, device=None):
        super(MyFasterRCNN, self).__init__()

        # 设备
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载预训练模型
        weights = fasterrcnn_resnet50_fpn(weights='DEFAULT') if pretrained else None
        self.model = fasterrcnn_resnet50_fpn(weights=weights)

        # 替换分类头
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # 移动到设备
        self.model.to(self.device)

    def forward(self, images, targets=None):
        # targets 可选：训练时传入，推理时可不传
        images = [img.to(self.device) for img in images]
        if targets is not None:
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        return self.model(images, targets)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # 推理模式


