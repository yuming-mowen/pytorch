import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import VGG16


# 加载测试数据
def test_data_process():
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                              download=True)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)
    return test_dataloader

# 测试代码
def test_model_process(model, test_dataloader):
    # 测试设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型到设备中
    model = model.to(device)

    # 初始化参数
    # 测试精度
    test_corrects = 0.0
    test_num = 0

    # 只进行前向传播
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            # 设置模型为评估模式
            model.eval()
            # 前向传播
            output = model(test_data_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则准确度test_corrects加1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 统计测试数据总数
            test_num += test_data_x.size(0)

        # 总精度
        test_acc = test_corrects.double().item() / test_num
        print("测试的准确率为：", test_acc)

# 推理代码
def reasoning_model_process(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义类别
    classes = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 模型设置为验证模式
            model.eval()
            # 进行模型推理
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = b_y.item()

            print("预测值：", classes[result], "---", "真实值：", classes[label])


# 开始测试
if __name__ == "__main__":
    # 加载模型
    model = VGG16()
    model.load_state_dict(torch.load('best_model.pth'))
    # 加载测试数据
    test_dataloader = test_data_process()

    # 加载模型测试的函数
    test_model_process(model, test_dataloader)
    # 模型推理结果
    reasoning_model_process(model, test_dataloader)

