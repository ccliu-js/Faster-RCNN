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
import sys
sys.stdout.reconfigure(line_buffering=True)
#数据集的文件位置
images_folder='dataset/images'
json_folder='dataset/processed_labels'


def collate_fn(batch):
    return tuple(zip(*batch))




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
        num_workers=4,### 多线程的开启之处
        collate_fn=collate_fn
    )





    # 模型部分
    num_classes = 6  # 5 个类别 + 背景
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)


    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ==== 3. 定义优化器 ====
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # ==== 4. 训练一个 epoch ====
    model.train()



    num_epochs = 1  # 设置训练轮数
    best_loss = float('inf')  # 初始为无穷大

    for epoch in range(num_epochs):
        print(f"=== Epoch {epoch+1}/{num_epochs} ===")
        model.train()
        epoch_loss = 0.0 # 累积每个epoch中每个batch的loss
        batch_index=0 # 


        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss+=losses.item() # 累加每个batch 的 loss

            batch_index+=1
            print(f'---------------batch: {batch_index}---------------')
            print(f"总损失: {losses.item():.4f}")
            print(f"分类损失: {loss_dict['loss_classifier'].item():.4f}, 边框回归损失: {loss_dict['loss_box_reg'].item():.4f}")
            print(f"RPN物体存在性损失: {loss_dict['loss_objectness'].item():.4f}, RPN边框回归损失: {loss_dict['loss_rpn_box_reg'].item():.4f}")



        avg_epoch_loss = epoch_loss / len(train_loader)  # 平均每个 epoch 的 loss
        print(f"Epoch {epoch+1} 平均损失: {avg_epoch_loss:.4f}")

            # 判断是否是最好的模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), 'weight/best_model.pth')
            print(f"保存新的最优模型，loss={best_loss:.4f}")











    # conv1_weights = model.backbone.body.conv1.weight.data.cpu()

    # # conv1_weights.shape => [64, 3, 7, 7] (out_channels, in_channels, kH, kW)

    # fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    # for i, ax in enumerate(axes.flatten()):
    #     # 将卷积核归一化到0~1
    #     kernel = conv1_weights[i]
    #     kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    #     # 转置通道为 HWC
    #     kernel = kernel.permute(1, 2, 0)
    #     ax.imshow(kernel)
    #     ax.axis('off')
    # plt.show()















    # ## 验证
    # model.eval()

    # for images, targets in val_loader:
    #     images = [img.to(device) for img in images]

    #     with torch.no_grad():
    #         outputs = model(images)  # 直接推理，不需要 targets

    #     # outputs 是一个列表，对应 batch 中每张图片
    #     for i in range(len(images)):
    #         img = images[i].cpu().permute(1, 2, 0).numpy()  # [C,H,W] -> [H,W,C] 并转为 numpy
    #         output = outputs[i]

    #         fig, ax = plt.subplots(1, figsize=(12,8))
    #         ax.imshow(img)

    #         # for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
    #         #     if score > 0.2:
    #         #         x1, y1, x2, y2 = box.cpu().numpy()
    #         #         rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
    #         #         ax.add_patch(rect)
    #         #         ax.text(x1, y1, f"{label.item()}:{score:.2f}", color='yellow', fontsize=12,
    #         #                 bbox=dict(facecolor='red', alpha=0.5))


    #         for box, score in zip(output['boxes'], output['scores']):
    #             if score > 0.4:  # 可根据需要调整阈值
    #                 x1, y1, x2, y2 = box.cpu().numpy()
    #                 rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
    #                                         linewidth=2, edgecolor='r', facecolor='none')
    #                 ax.add_patch(rect)


    #         plt.show()



    # input("press enter to exit ")

    