import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as f

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        # 网络参数
        self.ReLU = nn.ReLU()
        # 卷积层1 卷积核11×11×1×96 stride=4
        self.c1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        # 最大池化层2 池化感受野为3×3 stride=2
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 卷积层3 卷积核5×5×96×256 stride=1 padding=2
        self.c3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        # 最大池化层4 池化感受野为3×3 stride=2
        self.s4 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 卷积层5 卷积核为3×3×256×384 stride=1 padding=1，
        self.c5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        # 卷积层6 卷积核为3×3×384×384 stride=1 padding=1，
        self.c6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        # 卷积层7 I卷积核为3×3×384×256 stride=1 padding=1，
        self.c7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        # 最大池化层8 池化感受野为3×3 stride=2
        self.s8 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 平展层
        self.flatten = nn.Flatten()
        # 线性全连接层
        self.f10 = nn.Linear(9216, 4096)
        self.f11 = nn.Linear(4096, 4096)
        self.f12 = nn.Linear(4096, 10)

    # 模型前向传播
    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.s4(x)
        x = self.ReLU(self.c5(x))
        x = self.ReLU(self.c6(x))
        x = self.ReLU(self.c7(x))
        x = self.s8(x)

        x = self.flatten(x)
        x = self.ReLU(self.f10(x))
        x = f.dropout(x, 0.5)
        x = self.ReLU(self.f11(x))
        x = f.dropout(x, 0.5)
        x = self.f12(x)
        return x

if __name__ == "__main__" :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = AlexNet().to(device)
    print(summary(model, (1, 227, 227)))