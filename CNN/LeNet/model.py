import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        # 网络参数
        # 卷积层1 Input为28×28×1 output为28×28×6
        # 卷积核5×5×1×6 stride=1 padding=2
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sig = nn.Sigmoid()

        # 平均池化层2 Input为28×28×6 output为14×14×6
        # 池化感受野为2×2 stride=2
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 卷积层3 Input为14×14×6 output为10×10×16
        # 卷积核5×5×6×16 stride=1 padding=0
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # 平均池化层4 Input为10×10×16 output为5×5×16
        # 池化感受野为2×2 stride=2
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 平展层
        self.flatten = nn.Flatten()

        # 线性全连接层
        self.f5 = nn.Linear(400, 120)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)

    # 模型前向传播
    def forward(self, x):
        x = self.sig(self.c1(x))
        x = self.s2(x)
        x = self.sig(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.sig(self.f6(x))
        x = self.sig(self.f7(x))
        return x

if __name__ == "__main__" :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))