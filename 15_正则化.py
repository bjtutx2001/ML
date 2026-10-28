"""
正则化（Regularization）是机器学习中用于防止模型过拟合的一种技术。
    描述：正则化就是在模型原本的损失函数中增加一个惩罚项，限制模型参数过大或模型过于复杂，从而提高模型的泛化能力。
    公式：L(w)‘=L(w) + 惩罚项     惩罚项跟据正则化确定
    分类：
        1、L1正则化：惩罚项为惩罚系数*|w|(绝对值)，L1可以把某些特征削弱成0（直接剔除）
        2、L2正则化：惩罚项为惩罚系数*w^2，L2正则化只能削弱特征，实际开发常用L2
    惩罚系数：在模型训练时指定，惩罚系数越大，特征修改力度越大，要取合适值

"""
#导包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso,LassoCV      #L1正则化，L1交叉验证
from sklearn.linear_model import Ridge,RidgeCV      #L2正则化，L2交叉验证
from sklearn.metrics import mean_squared_error      #均方误差

#生成数据
np.random.seed(37)
x = np.random.uniform(low=-3,high=3,size=100)
y = 0.5 * x ** 2 + 2 * x + 1 + np.random.normal(0,1,size=100)

X=x.reshape(-1,1)
X=np.hstack((X,X**2,X**3,X**4,X**5,X**6,X**7,X**8,X**9,X**10))

#L1正则化
estimator_L1=Lasso(alpha=0.1)       #alpha为惩罚系数，默认为1
estimator_L1.fit(X,y)
y_pre_L1=estimator_L1.predict(X)

#L2正则化
estimator_L2=Ridge(alpha=0.1)
estimator_L2.fit(X,y)
y_pre_L2=estimator_L2.predict(X)

#常规线性回归
estimator=LinearRegression()
estimator.fit(X,y)
y_pre=estimator.predict(X)

#模型评估
print(f'L1正则化的均方误差：{mean_squared_error(y,y_pre_L1)}')
print(f'L2正则化的均方误差：{mean_squared_error(y,y_pre_L2)}')
print(f'普通回归的均方误差：{mean_squared_error(y,y_pre)}')



