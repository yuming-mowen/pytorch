import torch
import torch.nn as nn
from torchsummary import summary


class C3D(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # 卷积参数
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.bn4 = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.bn5 = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.f6 = nn.Linear(512*4*4, 4096)
        self.f7 = nn.Linear(4096, 4096)
        self.f8 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(p=0.5)

        # 初始化权重
        self.__init_weight()
        # 如果存在预训练权重，则加载预训练权重中的数值
        if pretrained:
            self.__load__pretrained_weights()

    # 前向传播
    def forward(self, x):
        # x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        # x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        # x = self.relu(self.bn3(self.conv3a(x)))
        # x = self.relu(self.bn3(self.conv3b(x)))
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        # x = self.relu(self.bn4(self.conv4a(x)))
        # x = self.relu(self.bn4(self.conv4b(x)))
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        # x = self.relu(self.bn5(self.conv5a(x)))
        # x = self.relu(self.bn5(self.conv5b(x)))
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.reshape(-1, 8192)  # 展平特征图
        x = self.relu(self.f6(x))
        x = self.dropout(x)
        x = self.relu(self.f7(x))
        x = self.dropout(x)
        x = self.f8(x)

        return x

    # 网络初始化
    def __init_weight(self):
        for m in self.modules():
            # 卷积层
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # 凯明初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 初始值为0
            # bn层
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # 全连接层
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 正态初始化 均值0 方差0.01
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 初始值为0

    # 加载预初始化权重
    def __load__pretrained_weights(self):
        # 定义对应关系字典
        corresp_name = {
            # Conv1
            "features.0.weight": "conv1.weight",
            "features.0.bias": "conv1.bias",
            # Conv2
            "features.3.weight": "conv2.weight",
            "features.3.bias": "conv2.bias",
            # Conv3a
            "features.6.weight": "conv3a.weight",
            "features.6.bias": "conv3a.bias",
            # Conv3b
            "features.8.weight": "conv3b.weight",
            "features.8.bias": "conv3b.bias",
            # Conv4a
            "features.11.weight": "conv4a.weight",
            "features.11.bias": "conv4a.bias",
            # Conv4b
            "features.13.weight": "conv4b.weight",
            "features.13.bias": "conv4b.bias",
            # Conv5a
            "features.16.weight": "conv5a.weight",
            "features.16.bias": "conv5a.bias",
            # Conv5b
            "features.18.weight": "conv5b.weight",
            "features.18.bias": "conv5b.bias",
            # f6
            "classifier.0.weight": "f6.weight",
            "classifier.0.bias": "f6.bias",
            # f7
            "classifier.3.weight": "f7.weight",
            "classifier.3.bias": "f7.bias",
        }

        # 加载权重
        p_dict = torch.load('ucf101-caffe.pth')  # 字典格式 包含网络层和对应参数
        s_dict = self.state_dict()

        # 不断循环，将预训练中corresp_name中的层的权重w和b赋值给我们搭建的C3D模型
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)  # 重新加载权重


if __name__ == "__main__" :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = C3D(num_classes=101).to(device)
    print(summary(model, (3, 16, 112, 112)))
