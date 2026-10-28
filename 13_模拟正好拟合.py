"""
演示正好拟合
"""

#导包
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,RidgeCV
import numpy as np

#定义函数，演示正好拟合
def dm02_just_fitting():
    #生成随机数据
    np.random.seed(37)
    x = np.random.uniform(low=-5,high=5,size=200)
    y = 0.5 * x ** 2 + 2 * x + 1 + np.random.normal(0,1,size=200)

    #数据预处理

    X=x.reshape(-1,1)
    # print(X**2)
    #线性回归的线性意思是指模型对待估计参数呈线性关系，而不是自变量与因变量呈现线性关系
    #增加平方项作为特征维度扩展
    X=np.hstack((X,X**2))   #hstack拼接的时候必须保证数组的维度数量一致，即二维拼二维，注意python的一维数组没有行列概念，维度写法为（m,）
    # print(X)

    #特征工程，这里特征就是X

    #模型训练
    estimator=LinearRegression()
    estimator.fit(X,y)

    #模型预测
    y_pre = estimator.predict(X)

    #模型评估
    print(f'均方误差为：{mean_squared_error(y,y_pre)}')

    plt.scatter(x,y)
    #直接绘图会由于点的顺序问题出现复杂曲线
    # plt.plot(x,y_pre,'r')
    index = np.argsort(x)   #argsort会返回排序的索引，默认升序
    index = index[::-1]     #降序写法，步长为负会改变切片方向，虽然区间是从左到右，但是切片方向是从右到左
    plt.plot(x[index],y_pre[index],color='red')
    plt.plot(np.sort(x),np.sort(y),color='red')
    plt.show()


#测试
if __name__ == '__main__':
    dm02_just_fitting()