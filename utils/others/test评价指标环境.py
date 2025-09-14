import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# 初始化 metric
metric = MeanAveragePrecision()

# 随机生成一个预测结果和目标，格式符合 COCO 格式
preds = [
    {
        "boxes": torch.tensor([[100, 100, 200, 200]]),  # [x1, y1, x2, y2]
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([1]),
    }
]

targets = [
    {
        "boxes": torch.tensor([[95, 95, 205, 205]]),
        "labels": torch.tensor([1]),
    }
]

# 更新 metric
metric.update(preds, targets)

# 计算结果
res = metric.compute()
print("\n📊 Mean Average Precision Results")
print("-" * 40)
for k, v in res.items():
    # 有些值是 tensor(-1.)，代表该项不适用
    if isinstance(v, torch.Tensor):
        val = v.item()
        if val == -1.0:
            print(f"{k:<20}: N/A")
        else:
            print(f"{k:<20}: {val:.4f}")
    else:
        print(f"{k:<20}: {v}")
