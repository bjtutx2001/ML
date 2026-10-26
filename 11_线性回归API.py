"""
案例：演示线性回归API使用，包括正规方程和随机梯度下降
数据源：波士顿房价预测
"""

#导包
# from sklearn.datasets import load_boston                #波士顿数据集
from sklearn.model_selection import train_test_split    #数据集划分
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error  # 均方误差
from sklearn.linear_model import LinearRegression       #正规方程线性回归
from sklearn.linear_model import SGDRegressor           #随机梯度下降
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.preprocessing import StandardScaler        #数据标准化

import pandas as pd
import numpy as np
#读数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  #hstack()水平拼接数据
target = raw_df.values[1::2, 2]

# print(f'数据特征{data.shape}')
# print(f'数据特征{target.shape}')
# print(f'数据特征{type(data)}')
# print(f'数据特征{data[0:5,:]}')

#数据预处理(数据集划分)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=37)

#特征工程（特征提取，特征预处理，这里是标准化）
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

#模型训练
#正规方程法
estimator_1=LinearRegression(fit_intercept=True)  #fit_intercept是是否需要截距，默认True
estimator_1.fit(x_train, y_train)

#梯度下降法
estimator_2=SGDRegressor(fit_intercept=True,learning_rate='constant',eta0=0.01,loss="squared_error")
estimator_2.fit(x_train, y_train)

#模型预测
y_pred_1=estimator_1.predict(x_test)
print(f'权重：{estimator_1.coef_}')
print(f'偏置：{estimator_1.intercept_}')
y_pred_2=estimator_2.predict(x_test)
print(f'权重：{estimator_2.coef_}')
print(f'偏置：{estimator_2.intercept_}')

#模型评估
print(f'正规方程均方误差：{mean_squared_error(y_test, y_pred_1)}')
print(f'正规方程均方根误差：{root_mean_squared_error(y_test, y_pred_1)}')
print(f'正规方程平均绝对误差：{mean_absolute_error(y_test, y_pred_1)}')
print(f'梯度下降均方误差：{mean_squared_error(y_test, y_pred_2)}')
print(f'梯度下降均方根误差：{root_mean_squared_error(y_test, y_pred_2)}')
print(f'梯度下降平均绝对误差：{mean_absolute_error(y_test, y_pred_2)}')

