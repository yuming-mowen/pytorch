import copy
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
import pandas as pd
from model import LeNet
import torch.nn as nn
import time

# 加载数据集
def train_val_data_process():
    # 处理训练集和验证集
    # 加载数据集
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)
    # 划分数据
    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])
    # 将训练集和数据集划分为 dataloader
    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=128,
                                   shuffle=True,
                                   num_workers=8)
    val_loader = Data.DataLoader(dataset=val_data,
                                 batch_size=128,
                                 shuffle=True,
                                 num_workers=8)

    return train_loader, val_loader

# 模型训练函数
def train_model_process(model, train_loader, val_loader, num_epochs):
    # 训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 定义学习率
    # 定义损失函数 交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 加载模型到设备中
    model = model.to(device)
    # 复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 损失值
    train_loss_all = []
    val_loss_all = []
    # 训练准确度
    train_acc_all = []
    val_acc_all = []

    # 训练时间
    since = time.time()

    # 开始训练
    for epoch in range(num_epochs):
        print("-" * 40)
        print("Epoch {}/{}".format(epoch + 1, num_epochs))

        # 初始化参数
        # 损失函数
        train_loss = 0.0
        val_loss = 0.0
        # 准确度
        train_corrects = 0.0
        val_corrects = 0.0
        # 样本数量
        train_num = 0
        val_num = 0

        # 取出数据进行训练
        for step, (b_x, b_y) in enumerate(train_loader):
            # 将数据放入设备中
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 打开模型训练模式
            model.train()
            # 进行前向传播 输入为一个batch,输出为一个batch的预测
            output = model(b_x)
            # 取最大概率对应的行标 对应该分类后的类别
            pre_lab = torch.argmax(output, dim=1)
            # 损失函数
            loss = criterion(output, b_y)

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 利用梯度下降法更新参数
            optimizer.step()

            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            '''
            loss.item() 为该批次平均值
            b_x.size(0) 该批次样本个数
            train_loss  最后的累加结果是该批次所有样本损失值的和
            '''
            # 计算精确度 如果预测正确，准确度加1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 统计训练集样本数量
            train_num += b_x.size(0)

        # 进行数据验证
        for step, (b_x, b_y) in enumerate(val_loader):
            # 将数据放入设备中
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 打开模型验证模式
            model.eval()
            # 进行前向传播 输入为一个batch,输出为一个batch的预测
            output = model(b_x)
            # 取最大概率对应的行标 对应该分类后的类别
            pre_lab = torch.argmax(output, dim=1)
            # 损失函数
            loss = criterion(output, b_y)

            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 计算精确度 如果预测正确，准确度加1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 统计验证集样本数量
            val_num += b_x.size(0)

        # 计算该轮次的损失值（平均值）
        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        # 计算该轮次的准确度（平均值）
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        # 打印该轮次相关参数
        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch + 1, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch + 1, val_loss_all[-1], val_acc_all[-1]))

        # 保存最优参数
        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前最高准确度的模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练和验证的耗时
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))
        time_pre = time_use * (num_epochs - epoch - 1) / (epoch + 1)
        print("预计还要{:.0f}m{:.0f}s".format(time_pre // 60, time_pre % 60))

    # 选择最优参数
    # 加载最高准确率的模型参数
    torch.save(best_model_wts, "./best_model.pth")
    # 将模型转化为dataframe格式
    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all, })
    return train_process

# 作图表示训练过程
def matplot_acc_loss(train_process):

    plt.figure(figsize=(12, 4))
    # 绘制loss图
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    # 绘制acc图
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, "bs-", label="Val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()

# 训练模型
if __name__ == "__main__":
    # 加载需要的模型
    LeNet = LeNet()
    # 加载数据集
    train_data, val_data = train_val_data_process()
    # 利用现有的模型进行模型的训练
    train_process = train_model_process(LeNet, train_data, val_data, num_epochs=50)
    matplot_acc_loss(train_process)