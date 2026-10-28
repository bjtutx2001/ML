"""
演示欠拟合
欠拟合结局方法：
    1、增加特征（多项式）
"""

#导包
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression   #正规方程
from sklearn.linear_model import Ridge,RidgeCV
import numpy as np
import matplotlib.pyplot as plt

#定义函数，模拟欠拟合
def dm01_under_fitting():
    #随机种子
    np.random.seed(37)
    #随机生成100个x轴数据，代表特征
    x=np.random.uniform(low=-3,high=3,size=100)     #最小值，最大值，个数
    #基于x值，根据线性公式（y=0.5x**2+2x+1+噪声），生成y数据，标签
    y=0.5 * x ** 2 + 2 * x + 1 + np.random.normal(0,1,size=100)
    # print(x)
    # print(y)
    plt.scatter(x,y)
    plt.xlabel("x")
    plt.ylabel("y")

    #数据预处理
    X=x.reshape(-1,1) #语法糖,训练数据需要是二维数组
    # Y=y.reshape(-1,1)
    # print(x.shape)
    # print(y.shape)

    #特征工程，略

    #模型训练
    estimator = LinearRegression()
    estimator.fit(X,y)

    #模型预测
    y_pred = estimator.predict(X)

    #模型评估
    print(f'均方误差：{mean_squared_error(y,y_pred)}')

    plt.plot(x,y_pred,color='red')
    plt.show()








#测试
if __name__=='__main__':
    dm01_under_fitting()
