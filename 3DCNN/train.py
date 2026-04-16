import torch

from C3D_model import C3D
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter
import os
from datetime import datetime
import socket
import timeit
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import VideoDataset

def train_model(num_classes, lr, device, save_dir, train_dataloader, val_dataloader, test_dataloader, num_epochs):
    '''
    :param num_classes: 分类的类别数量
    :param lr: 初始学习率
    :param device: 训练设备
    :param save_dir: 训练日志保存路径
    :param train_dataloader: 训练集数据
    :param val_dataloader: 验证集数据
    :param test_dataloader: 测试集数据
    :param num_epochs: 训练轮次
    :return:
    '''
    # 模型实例化
    model = C3D(num_classes, pretrained=True)
    # 定义损失函数 交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # 定义学习率更新参数 每10轮学习率*0.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 将模型放入设备中
    model.to(device)
    criterion.to(device)

    # 日志记录
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # 开始模型的训练
    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}  # 将验证集和训练集以字典的形式报存
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in
                      ['train', 'val']}  # 计算训练集和验证的大小 {'train': 8460, 'val': 2159}
    test_size = len(test_dataloader.dataset)  # 计算测试集的大小test_size:2701

    # 在训练循环之前添加变量来跟踪最佳验证损失
    best_val_loss = float('inf')
    best_epoch = -1
    best_model = model.state_dict()

    # 开始训练
    for epoch in range(num_epochs):
        for phase in['train', 'val']:
            start_time = timeit.default_timer()  # 计算训练开始时间
            running_loss = 0.0  # 初始化loss值
            running_corrects = 0.0  # 初始化准确率值
            # 选择是训练还是验证模式
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # 调用数据
            for inputs, labels in tqdm(trainval_loaders[phase]):
                # 将数据和标签放入到设备中
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()  # 清除梯度

                if phase == "train":
                    outputs = model(inputs)  # 前向传播
                else:
                    with torch.no_grad():  # 验证集不更新梯度
                        outputs = model(inputs)
                # 计算softmax的输出概率
                probs = nn.Softmax(dim=1)(outputs)
                # 计算最大概率值的标签
                preds = torch.max(probs, 1)[1]
                # 与标签对比计算损失函数
                labels = labels.long()
                loss = criterion(outputs, labels)
                # 训练时反向传播进行更新
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                # 计算训练过程中的相关量
                # 计算该轮次所有loss值的累加
                running_loss += loss.item() * inputs.size(0)
                # 计算该轮次所有预测正确值的累加
                running_corrects += torch.sum(preds == labels.data)

            scheduler.step()  # 对应进行学习率衰减

            epoch_loss = running_loss / trainval_sizes[phase]  # 计算该轮次的loss值，总loss除以样本数量
            epoch_acc = running_corrects.double() / trainval_sizes[phase]  # 计算该轮次的准确率值，总预测正确值除以样本数量
            # 将loss&acc写入writer
            if phase == "train":
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

                # 检查是否为最佳验证损失
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_epoch = epoch
                    best_model = model.state_dict()

            # 计算停止的时间戳
            stop_time = timeit.default_timer()

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, num_epochs, epoch_loss, epoch_acc))
            print("Execution time: " + str(stop_time - start_time) + "\n")
    writer.close()

    # 保存训练的好权重
    torch.save({
        'epoch': best_epoch + 1,
        'state_dict': best_model,
        'opt_dict': optimizer.state_dict(),
    },
        os.path.join(save_dir, 'models', 'C3D' + '_epoch-' + str(num_epochs) + '.pth.tar'))
    print("Save model at {}\n".format(os.path.join(save_dir, 'models', 'C3D' + '_epoch-' + str(num_epochs) + '.pth.tar')))

    # 开始模型的测试
    model.eval()
    running_corrects = 0.0  # 初始化准确率的值
    # 循环推理测试集中的数据，并计算准确率
    for inputs, labels in tqdm(test_dataloader):
        # 将数据和标签放入到设备中
        inputs = inputs.to(device)
        labels = labels.long()
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)

        # 计算softmax的输出概率
        probs = nn.Softmax(dim=1)(outputs)
        # 计算最大概率值的标签
        preds = torch.max(probs, 1)[1]

        running_corrects += torch.sum(preds == labels.data)
    epoch_acc = running_corrects.double() / test_size  # 计算该轮次的准确率值，总预测正确值除以样本数量
    print("test Acc: {}".format(epoch_acc))


if __name__ == "__main__":
    # 数据集初始化
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(VideoDataset(dataset_path='./data/ucf101', images_path='train', clip_len=16),
                                  batch_size=24,
                                  shuffle=True,
                                  num_workers=2)
    val_dataloader = DataLoader(VideoDataset(dataset_path='./data/ucf101', images_path='val', clip_len=16),
                                batch_size=24,
                                num_workers=2)
    test_dataloader = DataLoader(VideoDataset(dataset_path='./data/ucf101', images_path='test', clip_len=16),
                                 batch_size=24,
                                 num_workers=2)
    # 定义模型训练的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 25  # 训练轮次
    num_classes = 101  # 模型使用的数据集和网络最后一层输出参数
    lr = 1e-3  # 学习率
    save_dir = 'model_result'  # 保存路径

    train_model(num_classes, lr, device, save_dir, train_dataloader, val_dataloader, test_dataloader, num_epochs)


