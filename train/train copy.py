import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from myDataset import  MyDataset  # 假设你把 MyDataset 放在 dataset.py 里
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# ==== 2. 加载预训练模型 ====
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

#数据集的文件位置
images_folder='dataset/images'
json_folder='dataset/processed_labels'


def collate_fn(batch):
    return tuple(zip(*batch))






def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
    return total_loss / len(data_loader)


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    model.eval()
    total_loss = 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
    return total_loss / len(data_loader)


def main(train_loader, val_loader):
    num_classes = 6  # 5类 + 背景
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 训练多个 epoch
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss = evaluate_loss(model, val_loader, device)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")




















if __name__=='__main__':
    dataset=MyDataset(images_folder,json_folder)

    all_indices = list(range(len(dataset)))

    # 按 8:2 划分
    train_idx, val_idx = train_test_split(all_indices, test_size=0.2, random_state=42)

    # 创建子集--划分训练集和验证集
    train_dataset = MyDataset(images_folder, json_folder, indices=train_idx)
    val_dataset = MyDataset(images_folder, json_folder, indices=val_idx)


    #处理好数据的放入方式
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,   # 小目标/大分辨率，建议先用 2 或 4
        shuffle=True,
        num_workers=4,  # Windows 建议设小点，比如 0~2；Linux 可设大点
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # images, targets = next(iter(train_loader))
    # print(f"批量大小: {len(images)}")
    # print(f"image shape: {images[0].shape}")
    # print(f"targets keys: {targets[0].keys()}")
    # print(f"boxes shape: {targets[0]['boxes'].shape}")
    # print(f"labels: {targets[0]['labels']}")






    main(train_loader, val_loader)





    # # 模型部分
    # num_classes = 6  # 5 个类别 + 背景
    # weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    # model = fasterrcnn_resnet50_fpn(weights=weights)


    # # 替换分类头
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # # ==== 3. 定义优化器 ====
    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # # ==== 4. 训练一个 epoch ====
    # model.train()
    # for images, targets in train_loader:
    #     images = [img.to(device) for img in images]
    #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #     loss_dict = model(images, targets)
    #     losses = sum(loss for loss in loss_dict.values())

    #     optimizer.zero_grad()
    #     losses.backward()
    #     optimizer.step()

    #     print(f"loss: {losses.item():.4f}, details: {loss_dict}")


    input("press enter to exit ")

    