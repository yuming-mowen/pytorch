from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

# 马 ↔ 斑马 数据集加载类（CycleGAN）
class HorseZebraDataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform=None):
        # 斑马图像所在目录
        self.root_zebra = root_zebra
        # 马图像所在目录
        self.root_horse = root_horse
        # 数据增强 / 预处理方法（通常使用 Albumentations）
        self.transform = transform

        # 获取两个域中所有图像文件名
        self.zebra_images = os.listdir(root_zebra)
        self.horse_images = os.listdir(root_horse)

        # 数据集长度取两个域中样本数的最大值
        # 这是 CycleGAN 非配对数据加载的常见做法
        self.length_dataset = max(
            len(self.zebra_images),
            len(self.horse_images)
        )

        # 分别记录两个域的样本数量
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)

    def __len__(self):
        # 返回数据集长度
        return self.length_dataset

    def __getitem__(self, index):
        # 使用取模运算，保证在样本数量不一致时能够循环取样
        zebra_img = self.zebra_images[index % self.zebra_len]
        horse_img = self.horse_images[index % self.horse_len]

        # 构造完整的图像路径
        zebra_path = os.path.join(self.root_zebra, zebra_img)
        horse_path = os.path.join(self.root_horse, horse_img)

        # 读取图像并转换为 RGB，再转为 numpy 数组
        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        # 若提供了 transform，则对两个域的图像同时做数据增强
        # image 和 image0 是 Albumentations 中多输入的常见写法
        if self.transform:
            augmentations = self.transform(
                image=zebra_img,
                image0=horse_img
            )
            zebra_img = augmentations["image"]
            horse_img = augmentations["image0"]

        # 返回一对非配对图像：(斑马, 马)
        return zebra_img, horse_img
