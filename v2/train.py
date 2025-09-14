from _dataset import MyDataset
from __dataloader import MyDataLoader

from _model import FasterRCNNModel
from _train import FasterRCNNTrainer
from _model import FasterRCNNModel




# 数据集的位置
images_folder='dataset/images'
json_folder='dataset/processed_labels'

if __name__=='__main__':


    # 直接传类
    data_loader_module = MyDataLoader(
        dataset_class=MyDataset,
        images_folder=images_folder,
        json_folder=json_folder,
        batch_size=2,
        num_workers=2
    )

    train_loader, val_loader = data_loader_module.get_dataloaders()

    model=FasterRCNNModel(num_classes=6)

    trainer = FasterRCNNTrainer(model, lr=0.005, ckpt_path="weight/best_model.pth", grad_clip=5.0)

    trainer.load_checkpoint()  # 如果之前有保存的模型可以加载

    trainer.fit(train_loader, val_loader, epochs=10)

