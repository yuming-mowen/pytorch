import torch
from torch import nn
from torchsummary import summary

# 定义残差快
class Residual(nn.Module):

    def __init__(self, in_channels, num_channels, use_1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        self.relu = nn.ReLU()
    # 前向传播
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        # 是否使用1*1卷积
        if self.conv3:
            x = self.conv3(x)
        # 跳转链接
        y = self.relu(y+x)
        return y

class ResNet(nn.Module):

    def __init__(self, Residual):
        super().__init__()
        # 网络参数
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        )
        self.b2 = nn.Sequential(
            Residual(in_channels=64, num_channels=64, use_1conv=False, stride=1),
            Residual(in_channels=64, num_channels=64, use_1conv=False, stride=1)
        )
        self.b3 = nn.Sequential(
            Residual(in_channels=64, num_channels=128, use_1conv=True, stride=2),
            Residual(in_channels=128, num_channels=128, use_1conv=False, stride=1)
        )
        self.b4 = nn.Sequential(
            Residual(in_channels=128, num_channels=256, use_1conv=True, stride=2),
            Residual(in_channels=256, num_channels=256, use_1conv=False, stride=1)
        )
        self.b5 = nn.Sequential(
            Residual(in_channels=256, num_channels=512, use_1conv=True, stride=2),
            Residual(in_channels=512, num_channels=512, use_1conv=False, stride=1)
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

        # 模型初始化
        for m in self.modules():
            # 卷积层
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')  # 凯明初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 初始值为0
            # bn层
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # 全连接层
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 正态初始化 均值0 方差0.01
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 初始值为0


    # 模型前向传播
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x

if __name__ == "__main__" :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = ResNet(Residual).to(device)
    print(summary(model, (1, 224, 224)))