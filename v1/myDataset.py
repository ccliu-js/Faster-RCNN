import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class MyDataset(Dataset):
    """
    数据加载的类，继承pytorch中的dataset类
    """
    def __init__(self, images_folder, json_folder, transforms=None,indices=None):
        self.images_folder = images_folder
        self.json_files = [
            os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith(".json")
        ]

        # 如果传入 indices，只保留对应子集
        if indices is not None:
            self.json_files = [self.json_files[i] for i in indices]


        self.transforms = transforms

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        # 读取 JSON
        with open(self.json_files[idx], "r", encoding="utf-8") as f:
            data = json.load(f)

        # 读取图片
        img_path = os.path.join(self.images_folder, data["imagePath"])
        image = Image.open(img_path).convert("RGB")
        image = F.to_tensor(image)  # 转换为 [C,H,W] tensor, 值在 [0,1]


        # 处理 boxes 和 labels
        boxes = torch.tensor(data["boxes"], dtype=torch.float32)
        labels = torch.tensor(data["labels"], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}


        # 可选的 transforms
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

