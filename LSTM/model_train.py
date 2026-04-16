import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
import keras
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
# print(dataset)

# 划分训练集和验证集
train = dataset.iloc[0: int(len(a)*0.8), [0]]  # 0.8训练集
val = dataset.iloc[int(len(a)*0.8) : int(len(a)*0.9), [0]]
# print(train)
# print(val)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(train)
val = scaler.fit_transform(val)

"""
进行训练集数据特征和对应标签的划分，其中前面96个采样点中的负荷特征
来预测第97个点的电力负荷值。
X         Y
1-96  ->  97
2-97  ->  98
...
n-n+95->  n+96
...
"""
# 设置训练集的特征列表和对应标签列表
x_train = []
y_train = []
for i in np.arange(96, len(train)):
    x_train.append(train[i-96:i, :])
    y_train.append(train[i])
# 将训练集由list格式变为array格式
x_train, y_train = np.array(x_train), np.array(y_train)
# print(x_train.shape)
# print(x_train)

x_val = []
y_val = []
for i in np.arange(96, len(val)):
    x_val.append(val[i-96:i, :])
    y_val.append(val[i])
# 将训练集由list格式变为array格式
x_val, y_val = np.array(x_val), np.array(y_val)

# 神经网络搭建
model = Sequential()
model.add(LSTM(10, return_sequences=True, activation='relu'))
model.add(LSTM(15, return_sequences=False, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 对模型进行编译，选用Adam优化器，学习率为0.01
model.compile(optimizer=keras.optimizers.Adam(0.01), loss='mean_squared_error')

# 将训练集和测试集放入网络进行训练，每批次送入的数据为512个数据，一共训练30轮，将测试集样本放入到神经网络中测试其验证集的loss值
history = model.fit(x_train, y_train, batch_size=512, epochs=30, validation_data=(x_val, y_val))

# 保存训练好的模型
model.save('LSTM_model.h5')

# 绘制训练集和测试集的loss值对比图
# 创建一个大小为（12，8）的画布
plt.figure(figsize=(12, 8))
# 传入训练集的loss和验证集的loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
# 设置图的参数，设置图的名字
plt.title("LSTM神经网络loss值", fontsize=15)
# 设置xy轴的刻度值大小
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# 设置xy轴的标签
plt.ylabel('loss值', fontsize=15)
plt.xlabel('训练轮次', fontsize=15)
# 设置图例文字大小
plt.legend(fontsize=15)
plt.show()