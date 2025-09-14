import os
import json
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    """
    数据加载的类，继承pytorch中的dataset类
    """
    def __init__(self, images_folder, json_folder, transforms=None):
        self.images_folder = images_folder
        self.json_files = [
            os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith(".json")
        ]
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


        # 处理 boxes 和 labels
        boxes = torch.tensor(data["boxes"], dtype=torch.float32)
        labels = torch.tensor(data["labels"], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}


        # 可选的 transforms
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

if __name__=='__main__':

    #数据集的文件位置
    images_folder='dataset/images'
    json_folder='dataset/processed_labels'

    dataset = MyDataset(images_folder, json_folder)

    # 测试长度
    print(f"数据集大小: {len(dataset)}")

    # 测试单张数据
    image, target = dataset[1]
    print(f"图片类型: {type(image)}, 大小: {image.size if hasattr(image, 'size') else image.shape}")
    print(f"boxes: {target['boxes']}")
    print(f"labels: {target['labels']}")

    # 可视化一张图片和 bbox
    def show_image_with_boxes(img, target):
        # 将 PIL.Image 转为 numpy
        if isinstance(img, torch.Tensor):
            img = torchvision.transforms.ToPILImage()(img)
        plt.imshow(img)
        for box in target['boxes']:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                fill=False, color='red', linewidth=0.5)
            plt.gca().add_patch(rect)
        plt.show()

    show_image_with_boxes(image, target)


    # 测试 DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    for imgs, targets in dataloader:
        print(f"batch images: {len(imgs)}, batch targets: {len(targets)}")
        break

    input("please enter to exit")
