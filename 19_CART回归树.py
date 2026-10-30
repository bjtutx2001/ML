"""
CART回归树选择特征作为分类依据时采用平方误差（平方损失）
"""
#导包
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor


#准备数据
x_train=[0,1,2,3,4,5,6,7,8,9]
x_train=np.reshape(x_train,(-1,1))
print(x_train)
y_train=[5.56,5.7,5.91,6.5,6.8,7.05,8.9,9.13,10]

#模型训练
estimator_1=DecisionTreeRegressor(max_depth=1,random_state=37)
estimator_2=DecisionTreeRegressor(max_depth=2,random_state=37)

estimator_1.fit(x_train,y_train)
estimator_2.fit(x_train,y_train)