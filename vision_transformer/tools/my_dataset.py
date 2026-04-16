from PIL import Image
# Optional HEIC support via pillow-heif (install with: pip install pillow-heif)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

from torch.utils.data import Dataset, DataLoader
from typing import Sequence, Tuple, Any, Optional
import os
import random
import torch
from torchvision import transforms


class ViTDataSet(Dataset):
    """
    自定义分类数据集：返回 (image_tensor, label)
    - image_tensor: Tensor[3, H, W]（经过 transform 后，通常 H=W=224）
    - label: int（类别 id）
    """

    def __init__(self,
                 images_path: Sequence[str],
                 images_class: Sequence[int],
                 transform: Optional[Any] = None,
                 skip_broken: bool = True,
                 attempt_heic_rename: bool = True):
        """
        Args:
            images_path: 图片路径列表/序列，例如 [".../classA/1.jpg", ".../classB/2.png", ...]
            images_class: 每张图片对应的类别 id 列表/序列，例如 [0, 0, 2, 4, ...]
            transform: torchvision.transforms 的组合，用于图像预处理/增强（PIL -> Tensor）
            skip_broken: 如果为 True，则在初始化时检测每张图片能否被打开；若不能，会尝试用 `.heic` 后缀修复（可选重命名），仍不能打开则从样本中移除
            attempt_heic_rename: 当遇到无法识别的图片时，尝试将文件后缀改为 `.heic` 并再次打开（会尝试重命名文件）
        """
        # 数据一致性检查：路径数与标签数必须一致
        assert len(images_path) == len(images_class), \
            f"images_path and images_class length mismatch: {len(images_path)} vs {len(images_class)}"

        self.images_path = list(images_path)
        self.images_class = list(images_class)
        self.transform = transform
        self.attempt_heic_rename = attempt_heic_rename

        # 可选：在初始化时预检图片，避免 DataLoader worker 因单张损坏图片崩溃
        if skip_broken:
            removed = []
            renamed = []
            kept_paths = []
            kept_labels = []

            for p, lbl in zip(self.images_path, self.images_class):
                ok = False
                try:
                    # verify() 只读头部并检查文件是否为可识别图像，不会完整解码
                    with Image.open(p) as img:
                        img.verify()
                    ok = True
                    kept_paths.append(p)
                    kept_labels.append(lbl)
                except Exception:
                    # 尝试以 .heic 后缀修复（若用户启用了 attempt_heic_rename）
                    if attempt_heic_rename:
                        base, _ = os.path.splitext(p)
                        new_p = base + ".heic"

                        # 若 new_p 已存在，优先尝试打开它；否则尝试把原文件重命名为 new_p 再打开
                        tried_new = False
                        if os.path.exists(new_p):
                            try:
                                with Image.open(new_p) as img:
                                    img.verify()
                                kept_paths.append(new_p)
                                kept_labels.append(lbl)
                                ok = True
                                tried_new = True
                            except Exception:
                                tried_new = True

                        if not tried_new:
                            # 尝试重命名（注意：重命名失败时保留原文件）
                            try:
                                os.rename(p, new_p)
                                renamed.append((p, new_p))
                                try:
                                    with Image.open(new_p) as img:
                                        img.verify()
                                    kept_paths.append(new_p)
                                    kept_labels.append(lbl)
                                    ok = True
                                except Exception:
                                    # 重命名后依然打不开 -> 恢复原名
                                    try:
                                        os.rename(new_p, p)
                                        renamed.pop()
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                    if not ok:
                        removed.append(p)
                        # 跳过该图片，不加入 kept_paths

            # 替换为过滤后的列表
            self.images_path = kept_paths
            self.images_class = kept_labels

            if removed:
                print(f"[WARN] Removed {len(removed)} unreadable images (first examples: {removed[:5]})")
            if renamed:
                print(f"[INFO] Renamed {len(renamed)} files to .heic (attempted to fix mislabeled HEIC): {renamed[:5]}")

    def __len__(self) -> int:
        """返回数据集样本数（DataLoader 用它确定 epoch 的长度）"""
        return len(self.images_path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        根据索引 idx 取出一条样本：
        - 在读取失败时尝试用 `.heic` 修复（若启用），若仍失败则随机返回其它样本，避免 DataLoader 因单张损坏图崩溃。
        """
        attempts = 0
        max_attempts = 3
        cur_idx = int(idx)

        while attempts < max_attempts:
            img_path = self.images_path[cur_idx]
            label = int(self.images_class[cur_idx])
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    if self.transform is not None:
                        img = self.transform(img)
                return img, label
            except Exception:
                # 尝试用 .heic 修复（若在初始化时启用了该策略则保留行为）
                if getattr(self, "attempt_heic_rename", True):
                    base, _ = os.path.splitext(img_path)
                    new_p = base + ".heic"

                    # 若已经存在 new_p，优先尝试打开
                    if os.path.exists(new_p):
                        try:
                            with Image.open(new_p) as img:
                                img = img.convert("RGB")
                                if self.transform is not None:
                                    img = self.transform(img)
                            return img, label
                        except Exception:
                            pass

                    # 否则尝试重命名后打开（失败则恢复）
                    try:
                        os.rename(img_path, new_p)
                        try:
                            with Image.open(new_p) as img:
                                img = img.convert("RGB")
                                if self.transform is not None:
                                    img = self.transform(img)
                            # 更新 dataset 中的路径（后续访问使用新文件名）
                            self.images_path[cur_idx] = new_p
                            return img, label
                        except Exception:
                            # 恢复原名
                            try:
                                os.rename(new_p, img_path)
                            except Exception:
                                pass
                    except Exception:
                        pass

                # 最后策略：随机返回其它样本以保证训练继续
                attempts += 1
                if len(self.images_path) <= 1:
                    # 集合中只有当前这一张图，返回一个全零伪图以保证不崩溃
                    H = W = 224
                    fake = torch.zeros(3, H, W, dtype=torch.float32)
                    return fake, label
                cur_idx = random.randrange(0, len(self.images_path))

        # 三次尝试仍失败 -> 抛出错误
        raise RuntimeError(f"Failed to load image after retries: {self.images_path[cur_idx]}")

    @staticmethod
    def collate_fn(batch: Sequence[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        自定义 batch 组装函数（给 DataLoader 使用）

        batch: list/tuple，长度=batch_size
               每个元素是 (img_tensor, label)

        返回：
          - images: Tensor[B, 3, H, W]
          - labels: Tensor[B]（dtype=torch.long，适配 CrossEntropyLoss）
        """
        # 将 [(img1,l1),(img2,l2),...] 拆成 ([img1,img2,...], [l1,l2,...])
        images, labels = tuple(zip(*batch))

        # stack 要求每张图尺寸一致（你的 transform 已裁剪到固定 img_size，因此成立）
        images = torch.stack(images, dim=0)

        # CrossEntropyLoss 需要 labels 是 int64(LongTensor)
        labels = torch.as_tensor(labels, dtype=torch.long)

        return images, labels

def build_vit_dataloaders(train_images_path, train_images_label,
                          val_images_path, val_images_label,
                          batch_size, img_size=224,
                          mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5),
                          num_workers=None):
    """
    构建 ViT 训练/验证的 DataLoader。

    输入：
      - train_images_path / val_images_path: 图片文件路径列表（list[str]）
      - train_images_label / val_images_label: 对应标签列表（list[int]）
      - batch_size: 每个 batch 的样本数
      - img_size: 模型输入分辨率（ViT-B/16 默认224）
      - mean/std: Normalize的均值/方差（这里是把 [0,1] 线性映射到近似 [-1,1]）
      - num_workers: DataLoader 多进程加载数据的 worker 数，None 则自动估算

    输出：
      - train_loader, val_loader
    """

    # 定义训练/验证的数据预处理与增强（torchvision.transforms）
    #    - train：随机裁剪 + 翻转（增强，提高泛化）
    #    - val：固定resize + center crop（保证评估稳定可复现）
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),   # 随机裁剪到 img_size，并随机缩放
            transforms.RandomHorizontalFlip(),        # 随机水平翻转
            transforms.ToTensor(),                    # PIL -> Tensor，范围变为 [0,1]
            transforms.Normalize(mean, std),          # 归一化（常见设置 mean=std=0.5）
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),                   # 先把短边缩放到 256（经典 ImageNet eval 流程）
            transforms.CenterCrop(img_size),          # 再中心裁剪到 img_size
            transforms.ToTensor(),                    # PIL -> Tensor，[0,1]
            transforms.Normalize(mean, std),          # 同样归一化
        ]),
    }

    # 用自定义Dataset封装“读图 + label + transform”
    # Dataset 的 __getitem__ 返回：(img_tensor, label)
    train_dataset = ViTDataSet(
        train_images_path, train_images_label,
        transform=data_transform["train"])
    val_dataset = ViTDataSet(
        val_images_path, val_images_label,
        transform=data_transform["val"])

    # 自动设置num_workers（加载数据的进程数）
    #  经验策略：不超过 CPU 核数、不超过 8，也不超过 batch_size（避免过多进程反而开销大）
    if num_workers is None:
        num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    # 训练集 DataLoader
    # - shuffle=True：每个 epoch 打乱数据顺序
    # - pin_memory=True：若用GPU，可加速 CPU->GPU 拷贝
    # - collate_fn：自定义 batch 组装方式（通常用于处理不同尺寸/额外信息）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn
    )

    # 验证集 DataLoader
    # - shuffle=False：验证/测试不需要打乱，保证评估稳定
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=val_dataset.collate_fn
    )

    # 返回两个loader，供训练循环使用
    return train_loader, val_loader

