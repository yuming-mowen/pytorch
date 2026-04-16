# 导入库
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 加载历史数据文件
# index_col=[0]，将第一列作为索引
dataset = pd.read_csv('load.csv', index_col=[0])
dataset = dataset.ffill()  # 数据填充
# print(dataset)


dataset = np.array(dataset)
# print(dataset)

# 将所有的数据放到一个列表里面，方便后续的训练集和测试集的制作
a = []
for item in dataset:
    for i in item:
        a.append(i)
dataset = pd.DataFrame(a)

real = np.array(dataset)
# print(real)


# 绘制总体数据图
# 创建一个大小为（20，8）的画布
plt.figure(figsize=(20, 8))
# 传入预测值和真实值
plt.plot(real)
# 设置xy轴的刻度值大小
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
labels = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
plt.xticks(range(0, 35040, 2920), labels=labels)
# 设置xy轴的标签
plt.ylabel('负荷值', fontsize=15)
plt.xlabel('月份', fontsize=15)
# 设置图的参数，设置图的名字
plt.title("数据总览", fontsize=15)
plt.show()


week_data = dataset.iloc[96*6:96*12, :]  # 1月6日开始
week_data = np.array(week_data)
# 绘制一周之内的变化图
# 创建一个大小为（20，8）的画布
plt.figure(figsize=(20, 8))
# 传入预测值和真实值
plt.plot(week_data)
# 设置xy轴的刻度值大小
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
plt.xticks(range(0, 96*7, 96), labels=labels)
# 设置xy轴的标签
plt.ylabel('负荷值', fontsize=15)
plt.xlabel('日期', fontsize=15)
# 设置图的参数，设置图的名字
plt.title("周数据", fontsize=15)
plt.show()