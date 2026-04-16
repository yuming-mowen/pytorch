import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入数据
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

# 划分测试集
test = dataset.iloc[int(len(a)*0.98):, [0]]
# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
test = scaler.fit_transform(test)

# 特征划分
x_test = []
y_test = []
for i in np.arange(96, len(test)):
    x_test.append(test[i-96:i, :])
    y_test.append(test[i])
# 将测试集由list格式变为array格式
x_test, y_test = np.array(x_test), np.array(y_test)

# 导入模型
model = load_model('LSTM_model.h5')
# 利用模型进行测试
y_predict = model.predict(x_test)
# 将真实值标签进行反归一化操作
real = scaler.inverse_transform(y_test)
# 将模型预测出的值进行反归一化操作
prediction = scaler.inverse_transform(y_predict)
# print(prediction)

# 绘制真实值和预测值对比图
# 创建一个大小为（12，8）的画布
plt.figure(figsize=(12, 8))
# 传入预测值和真实值
plt.plot(prediction, label='预测值')
plt.plot(real, label='真实值')
# 设置xy轴的刻度值大小
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# 设置xy轴的标签
plt.legend(loc='best', fontsize=15)
plt.ylabel('负荷值', fontsize=15)
plt.xlabel('采样点', fontsize=15)
# 设置图的参数，设置图的名字
plt.title("基于LSTM神经网络负荷预测", fontsize=15)
plt.show()


# 调用模型评价指标
# R2
from sklearn.metrics import r2_score
# MSE
from sklearn.metrics import mean_squared_error
# MAE
from sklearn.metrics import mean_absolute_error
# 计算模型的评价指标
R2 = r2_score(real, prediction)
MAE = mean_absolute_error(real, prediction)
RMSE = np.sqrt(mean_squared_error(real, prediction))
MAPE = np.mean(np.abs((real-prediction) / prediction))
# 打印模型的评价指标
print('R2:', R2)
print('MAE:', MAE)
print('RMSE:', RMSE)
print('MAPE:', MAPE)