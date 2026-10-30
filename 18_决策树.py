"""
决策树：
    概述：决策树是一种树形结构，由三部分组成：根节点、内部节点，叶节点，
        其核心思想是通过选择合适的特征和划分条件，将数据集逐步划分为不同的子集，最终形成树状结构，实现预测。
        即能回归也能分类，属于监督学习（有特征，有标签）
    常见决策树算法：
        1、ID3决策树：以信息增益作为特征选则的依据
            信息增益：信息熵——条件熵
        2、C4.5决策树：以信息增益率作为特征选择依据
            信息增益率：信息增益/特征熵（划分信息）
        3、CART决策树：以划分后的加权基尼指数作为分类树选择条件，以平方误差作为回归数选择条件
            基尼指数：1-特征概率平方和，基尼指数越小，数据越纯净
        信息增益和信息增益率越大越好，划分后的加权基尼指数越小越好

案例：使用决策数模块演示要泰坦尼克号CART树分类
"""
#导包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

#加载数据
train=pd.read_csv("./data/titanic_train.csv")
test=pd.read_csv("./data/titanic_test.csv")
# print(train.head())

#数据预处理
#发现age项有缺失，用该列的平均值做补全
train['Age']=train['Age'].fillna(train['Age'].mean())
test['Age']=test['Age'].fillna(test['Age'].mean())
#对Sex列进行one-hot编码，独热编码
train=pd.get_dummies(train,columns=['Sex'])
test=pd.get_dummies(test,columns=['Sex'])
train=train.drop(columns=['Sex_male'])
test=test.drop(columns=['Sex_male'])

#数据集划分已完成

#特征工程
# 特征提取
#训练集
x=train[['Pclass','Sex_female','Age','Parch']]
y=train['Survived']
#测试集，没有Survived
x_test_1=test[['Pclass','Sex_female','Age','Parch']]
# y_test=test['Survived']

# 数据集划分
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=37)
X_train=x_train
print(x_train)
#标准化
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train) #标准化会将DF转成ndarray
x_test=transfer.transform(x_test)
x_test_1=transfer.transform(x_test_1)

#模型训练
estimator=DecisionTreeClassifier(max_depth=10,random_state=37)
estimator=estimator.fit(x_train,y_train)

#模型预测
y_pred=estimator.predict(x_test)
y_pred_1=estimator.predict(x_test_1)

#模型评估
print(f'模型分类评估报告：\n{classification_report(y_test,y_pred)}')
print(f'测试集评估结果：{y_pred_1}')
# print(x_train)

#绘制决策树
plt.figure(figsize=(30,20),dpi=80)  #30*20英寸，DPi表示每英寸的像素个数，结果为30*80,20*80的像素尺寸图片
#参1：模型对象，参2：填充节点颜色，参3：使用训练集的列名作为节点划分条件中的特征名称，参4：数深度
plot_tree(estimator,filled=True,feature_names=X_train.columns,max_depth=10)
plt.savefig('./data/my_titanic.png')
plt.show()

# print(train.info())
# print(test.info())
# print(train.columns)
# print(test.columns)
