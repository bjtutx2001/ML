"""
演示过拟合
解决方法：
    1、清洗数据
    2、增大数据训练量
    3、正则化
    4、减少特征维度

"""
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,RidgeCV
import numpy as np


def dm03_upper_fitting():
    np.random.seed(37)
    x = np.random.uniform(low=-5, high=5, size=100)
    y = 0.5 * x ** 2 + 2 * x + 3 + np.random.normal(0,1,size=100)

    X=x.reshape(-1,1)
    #扩展多个特征，使模型变复杂
    X=np.hstack((X,X**2,X**3,X**4,X**5,X**6,X**7,X**8,X**9,X**10))
    # print(X)

    estimator = LinearRegression()
    estimator.fit(X,y)
    y_pred = estimator.predict(X)

    print(f'均方误差：{mean_squared_error(y,y_pred)}')

    plt.scatter(x,y)
    index=x.argsort()
    plt.plot(x[index],y_pred[index],color='red')
    plt.show()



#测试
if __name__ == '__main__':
    dm03_upper_fitting()