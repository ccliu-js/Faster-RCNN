from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class MyDataLoader:
    """
    自定义 DataLoader 封装类
    功能：
    - 自动划分训练集/验证集
    - 自动创建 train_loader 和 val_loader
    - 内部封装 collate_fn
    """
    def __init__(self, dataset_class, images_folder, json_folder, batch_size=2, 
                 num_workers=4, test_size=0.2, random_state=42, transforms=None):
        self.dataset_class = dataset_class
        self.images_folder = images_folder
        self.json_folder = json_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.random_state = random_state
        self.transforms = transforms

        # 初始化：划分训练/验证索引
        full_dataset = self.dataset_class(images_folder, json_folder, transforms=transforms)
        all_indices = list(range(len(full_dataset)))
        self.train_idx, self.val_idx = train_test_split(all_indices, test_size=test_size, random_state=random_state)

    # =========================
    # 内部 collate_fn
    # =========================
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    # =========================
    # 获取 Dataset  -->先获取到 dataset，实现数据格式的正确处理，此后才是数据的划分 和 其他处理
    # =========================
    def get_datasets(self):
        train_dataset = self.dataset_class(self.images_folder, self.json_folder, 
                                           transforms=self.transforms, indices=self.train_idx)
        val_dataset = self.dataset_class(self.images_folder, self.json_folder, 
                                         transforms=self.transforms, indices=self.val_idx)
        return train_dataset, val_dataset

    # =========================
    # 获取 DataLoader
    # =========================
    def get_dataloaders(self):
        train_dataset, val_dataset = self.get_datasets()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

        return train_loader, val_loader
