import torch
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class FasterRCNNTrainer:
    def __init__(self, model, lr=0.005, ckpt_path="weight/best_model.pth", device=None, grad_clip=None):
        """
        grad_clip: 
            float or None, 如果不为 None，则对梯度进行裁剪
            梯度裁剪，限制梯度的最大值，防止梯度不稳定 或者 loss爆炸。
            通俗点讲：限制梯度的更新的幅度，降低参数跳变的幅度（可以从学习率这种类似的思想去理解）

            使用原因：
                1.在小 batch 或大 loss 的情况下，梯度可能会非常大（梯度爆炸）
                2.直接用梯度更新参数，会导致模型参数变化过大 → loss NaN 或训练不稳定
                3.裁剪梯度可以限制每次更新的最大幅度，使训练更加稳定

        """
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)

        self.ckpt_path = ckpt_path
        self.best_loss = float("inf")###用于暂存 当前模型的loss的，便于后续的各种分析和处理
        self.grad_clip = grad_clip

    def train_one_epoch(self, data_loader, epoch):
        self.model.train()
        total_loss = 0

        loop = tqdm(data_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for images, targets in loop:
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            total_loss += losses.item()

            loop.set_postfix(loss=losses.item())

        avg_loss = total_loss / len(data_loader)
        print(f"[Epoch {epoch}] Training Loss: {avg_loss:.4f}")
        return avg_loss

 
    def validate(self, data_loader, epoch):
        self.model.eval()
        device = self.device   # 假设你在 Trainer 初始化时就有 self.device
        all_predictions = []
        all_targets = []

        loop = tqdm(data_loader, desc=f"Epoch {epoch} [Val]", leave=False)

        with torch.no_grad():
            for images, targets in loop:
                # 把每张图像搬到 GPU
                images = [img.to(device) for img in images]

                # 把 targets 的每个 dict 搬到 GPU
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # 推理
                predictions = self.model(images)

                # 确保预测结果也在 GPU
                predictions = [{k: v.to(device) for k, v in p.items()} for p in predictions]

                # 收集结果
                all_predictions.extend(predictions)
                all_targets.extend(targets)

        # 初始化评估器
        metric = MeanAveragePrecision(class_metrics=True)
        # 更新结果
        metric.update(all_predictions, all_targets)
        # 计算最终结果
        res = metric.compute()

        print(f"\n📊 [Epoch {epoch}] - Validation Mean Average Precision Results")
        print("-" * 40)
        for k, v in res.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:  # 只有一个元素，可以安全转成 float
                    val = v.item()
                    if val == -1.0:
                        print(f"{k:<20}: N/A")
                    else:
                        print(f"{k:<20}: {val:.4f}")
                else:
                    # 多类别的情况，打印前几个，或者全部
                    print(f"{k:<20}: {v.tolist()}")
            else:
                print(f"{k:<20}: {v}")

        return res["map"].item()




    def save_checkpoint(self):
        torch.save(self.model.state_dict(), self.ckpt_path)

    def load_checkpoint(self):
        try:
            self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
            print(f"🔄 Loaded checkpoint from {self.ckpt_path}")
        except FileNotFoundError:
            print("⚠️ No checkpoint found, starting from scratch.")

    # =========================
    # 新增 fit 方法
    # =========================
    def fit(self, train_loader, val_loader, epochs):
        # 储存训练过程中的数据，便于后续进行分析训练过程 和 做好可视化
        best_train_loss = float("inf")  # 初始化最好的训练损失为无穷大
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)

            # 如果当前训练损失优于历史最佳损失，则保存模型
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                # self.save_checkpoint()  # 保存当前模型
                self.validate(val_loader, epoch)

            self.lr_scheduler.step()## 这个是什么？？会有什么用处？？
