"""
AdaBoost算法：
    介绍：属于Boosting思想，串行执行，每次使用全部样本，最后加权投票
    原理：
        1、使用全部样本，通过决策树进行模型训练，获取结果
            思路：预测成功，样本权重下降
                 预测失败，样本权重提升
        2、把第一个弱分类器的结果交给第二个弱分类器进行训练
        3、以此类推
    大致过程：初始化样本权重——>训练第一个学习器——>计算学习器权重——>更新样本权重——>根据划分后的基尼指数计算第二个学习器——>组合学习器
案例：演示葡萄酒分类预测
"""

#导包
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder          #编码编译器
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#导入数据
data=pd.read_csv('./data/红酒品质分类.csv')
# print(data.head())
#二分类，保留两种品质的酒
data=data[data['quality'].isin([5,6])]
print(data.info())
print(data['quality'].value_counts())
#labelencoder，编码编译器主要用于目标标签 y，把不同类别编码为整数。如猫、狗，这里已经是数值，可以不转，做演示
encoder=LabelEncoder()
y=encoder.fit_transform(data['quality'])
# print(y)
#获取特征
x=data[['fixed acidity','volatile acidity','density','sulphates','alcohol',]]

#分割数据
#stratify分层抽样参数用于控制训练集与测试集中样本的比例，和特征y一样，防止某个种类样本占比 过大
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=37,stratify=y)

#特征工程，此处略

#模型训练
#普通决策树
estimator_1=DecisionTreeClassifier(max_depth=3,random_state=37)
estimator_1.fit(x_train,y_train)
#Adaboost
#参数1：传入的弱分类器对象，参数2：弱分类器数量，参数3：学习率，乘以（影响）分类器权重的系数，学习率越大，每轮调整越激进，预测错误样本的相对权重增加得越明显；学习率越小，每轮调整越温和。 参数4：随机种子
#注意传入的弱分类器只是借用分类器模板，可以是训练过的，系统会自动复制一个参数一样的没训练过的
estimator_2=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3),n_estimators=100,learning_rate=0.1,random_state=37)
estimator_2.fit(x_train,y_train)

#模型预测
y_pre_1=estimator_1.predict(x_test)
y_pre_2=estimator_2.predict(x_test)

#模型评估
print(f'决策树预测准确率：{estimator_1.score(x_test,y_test)}')
print(f'AdaBoost预测准确率：{accuracy_score(y_test,y_pre_2)}')
