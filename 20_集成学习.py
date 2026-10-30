"""
集成学习：
    概述：将多个模型组合起来，共同完成预测任务，预测结果比单个模型更加稳定和准确
    分类：
        1、Bagging(装袋法)：并行训练多个模型，汇总预测结果（平权投票表决），例如Random Forest随机森林
        2、Boosting(提升法)：按顺序训练多个模型，后续模型重点改进前面模型存在的预测问题，最后将多个模型组合起来，加权投票表决，例如AdaBoost
        3、Stacking(堆叠法)：先训练多个模型，在训练一个新的模型，学习如何组合这些基础模型的预测结果
    Bagging之决策树：决策树的核心思想是训练多棵CART树，综合这些树的训练结果，既可以回归也可以分类
        训练集选择：进行有放回的抽样，默认情况下每棵树抽取的样本数量与原始数据集相同，但不必包含全部原始数据集，因此可能存在重复样本
            自助采样：
                1、每棵树的数据可能不同
                2、有放回的抽样，同一个样本可能被重复抽中
                3、默认情况下每棵树抽取的样本数量与原始数据集相同，但不必包含全部原始数据集
            弱学习器的训练样本既有交集也有差异，更容易发挥投票的作用
        特征选择：每棵树随机选择n个特征作为待选取特征，在选中的候选特征内部，CART决策树仍然使用基尼指数等准则寻找最优划分。
        要注意，最终的预测结果有两种预测方式，分别为硬投票和软投票
            硬投票：根据预测结果进行多数表决，需要手动实现（直接进行弱学习器分类汇总）
            软投票：根据预测概率计算分类综合概率，选概率最大的类别（默认）
"""
#案例：演示随机森林算法，使用泰坦尼克号数据集

#导包
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split    #数据计划
from sklearn.ensemble import RandomForestClassifier     #随机森林
from sklearn.model_selection import GridSearchCV        #网格搜索

#准备数据
train=pd.read_csv('./data/titanic_train.csv')
test=pd.read_csv('./data/titanic_test.csv')

# print(train.info())

#填充Age空白值
train['Age']=train['Age'].fillna(train['Age'].mean())
test['Age']=test['Age'].fillna(test['Age'].mean())

#抽取特征
x=train[['Pclass','Sex','Age','SibSp','Fare']]
y=train['Survived']

#Sex进行独热编码
x=pd.get_dummies(x,columns=['Sex'])
x=x.drop(columns=['Sex_male'])
print(x.info())

#划分数据集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=37)

#模型训练
#单一决策树
estimator_1=DecisionTreeClassifier(max_depth=10,random_state=37)    #random_state控制决策树结构，如果恰好有两个特征的基尼指数相同，随机数就能控制每次的先后顺序
estimator_1.fit(x_train,y_train)
#随机森林，默认参数
estimator_2=RandomForestClassifier()
estimator_2.fit(x_train,y_train)
#随机森林，网格搜索
estimator_3=RandomForestClassifier()
#网格搜索参数,字典形式
params={'n_estimators':[30,60,90,120,150],'max_depth':[3,5,7,9]}
#网格搜索学习器创建
estimator_3=GridSearchCV(estimator=estimator_3,param_grid=params,cv=5)
estimator_3.fit(x_train,y_train)

#模型预测
y_pre_1=estimator_1.predict(x_test)
y_pre_2=estimator_2.predict(x_test)
y_pre_3=estimator_3.predict(x_test)

#模型评估
print(f'单棵决策树的预测准确率为：{estimator_1.score(x_test,y_test)}')
print(f'单棵决策树的预测准确率为：{accuracy_score(y_test,y_pre_1)}')
print(f'随色森林的预测准确率为{accuracy_score(y_test,y_pre_2)}')
print(f'随色森林+网格搜索的预测准确率为{accuracy_score(y_test,y_pre_3)}')
print(f'获取最佳参数：{estimator_3.best_score_}')
print(f'获取最佳参数：{estimator_3.best_params_}')







