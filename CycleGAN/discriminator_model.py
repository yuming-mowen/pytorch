import torch
import torch.nn as nn

# 判别器中使用的基础卷积模块
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        # 一个典型的 CycleGAN 判别器卷积块：
        # Conv2d + InstanceNorm + LeakyReLU
        self.conv = nn.Sequential(
            # 4×4 卷积核是 PatchGAN 判别器的常见设置
            # stride 决定是否进行下采样
            # padding_mode="reflect" 可减少边界伪影
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode="reflect"
            ),
            # Instance Normalization，适合风格迁移 / 图像生成任务
            nn.InstanceNorm2d(out_channels),
            # LeakyReLU 激活函数，负半轴保留 0.2 的斜率
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        # 前向传播：直接通过该卷积块
        return self.conv(x)

# CycleGAN 的判别器（PatchGAN）
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        # 初始卷积层：
        # 不使用归一化，只做一次下采样
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,        # 输入图像通道数（RGB 为 3）
                features[0],        # 第一层特征图通道数
                kernel_size=4,
                stride=2,           # 下采样
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        # 构建后续卷积层
        layers = []
        in_channels = features[0]

        for feature in features[1:]:
            # 最后一层特征（512）不再下采样，stride=1
            # 其余层使用 stride=2 进行下采样
            layers.append(
                Block(
                    in_channels,
                    feature,
                    stride=1 if feature == features[-1] else 2
                )
            )
            in_channels = feature

        # 最后一层 1×输出通道的卷积
        # 输出的是 PatchGAN 的判别结果（特征图形式）
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect"
            )
        )

        # 将所有层打包成一个 Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # 先通过初始卷积层
        x = self.initial(x)
        # 再通过主干网络，并用 sigmoid 映射到 [0, 1]
        # 表示每个 patch 为真实图像的概率
        return torch.sigmoid(self.model(x))

# 简单测试函数
def test():
    # 构造一个 256×256 的随机 RGB 输入
    x = torch.randn((1, 3, 256, 256))
    # 实例化判别器
    model = Discriminator(in_channels=3)
    # 前向传播
    preds = model(x)
    # 打印模型结构
    print(model)
    # 打印输出尺寸（PatchGAN 输出特征图大小）
    print(preds.shape)

if __name__ == "__main__":
    test()
