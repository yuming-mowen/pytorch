import torch
import torch.nn as nn

# 卷积 / 反卷积基础模块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        # 根据 down 参数决定使用普通卷积（下采样）还是反卷积（上采样）
        self.conv = nn.Sequential(
            # down=True：使用 Conv2d 进行下采样或特征提取
            # down=False：使用 ConvTranspose2d 进行上采样
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            # InstanceNorm 在生成模型中比 BatchNorm 更稳定
            nn.InstanceNorm2d(out_channels),
            # 是否使用 ReLU 激活函数
            # 在某些层（如残差块的最后一层）中不使用激活
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        # 前向传播：直接通过该卷积块
        return self.conv(x)

# 残差块（Residual Block）
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 两个 3×3 卷积组成的残差结构
        self.block = nn.Sequential(
            # 第一个卷积：带 ReLU
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            # 第二个卷积：不使用激活函数
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # 残差连接：输入与卷积结果相加
        return x + self.block(x)

# CycleGAN 的生成器（ResNet-based）
class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()

        # 初始卷积层：
        # 使用 7×7 卷积提取低层特征，保持分辨率不变
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect"
            ),
            nn.ReLU(inplace=True),
        )
        # 下采样模块：
        # 通过 stride=2 的卷积逐步减小空间尺寸、增加通道数
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features,
                    num_features * 2,
                    down=True,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 4,
                    down=True,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
            ]
        )
        # 残差块堆叠：
        # 在低分辨率、高通道特征空间中进行非线性变换
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)],
        )
        # 上采样模块：
        # 使用反卷积逐步恢复空间分辨率
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features * 4,
                    num_features * 2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 1,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
            ]
        )
        # 输出层：
        # 使用 7×7 卷积将特征映射回图像通道数
        self.last = nn.Conv2d(
            num_features,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect"
        )

    def forward(self, x):
        # 初始特征提取
        x = self.initial(x)
        # 下采样
        for layer in self.down_blocks:
            x = layer(x)
        # 残差变换
        x = self.residual_blocks(x)
        # 上采样
        for layer in self.up_blocks:
            x = layer(x)

        # 使用 tanh 将输出限制在 [-1, 1]，符合 CycleGAN 的图像归一化约定
        return torch.tanh(self.last(x))

# 简单测试函数
def test():
    image_channels = 3
    img_size = 256
    # 构造一个 batch=2 的随机输入
    x = torch.randn((2, image_channels, img_size, img_size))
    # 实例化生成器
    gen = Generator(image_channels, 9)
    # 打印模型结构
    print(gen)
    # 打印输出尺寸
    print(gen(x).shape)

if __name__ == "__main__":
    test()
