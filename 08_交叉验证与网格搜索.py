"""
交叉验证：
    一种数据分割方法，将数据分为n份，拿一份作为验证集，其他n-1份作为训练集
    原理：
        第一次去第一份做验证，剩下的做训练
        第二次去第二份做验证，剩下的做训练
        以此类推，总共得到n次训练结果，取n次结果的平均作为训练结果
        若第k次的模型结果最好，则最后用所有数据（训练集+验证集）对第k次的模型进行训练，在使用第k次的验证集进行验证
    交叉验证解决的是模型的数据输入问题（数据集划分）

网格搜索：
    超参数：超参数（Hyperparameter）是机器学习模型在训练之前，由开发者预先设置的参数，用于控制模型的学习过程、结构或预测行为。
    例如KNN算法中的K，一般模型有很多超参数，需要手动设置很多超参数组合来获取最优模型，每组超参数都需要进行交叉验证评估
    而网格搜索则是模型调参的有效工具，只需要将若干参数传给网格搜索对象，他会帮我们完成不同超参数的组合、训练和评估，最终返回最优超参数
    网格搜索解决的是超参数的组合问题

网格搜索+交叉验证的强力组合可以形成一个模型参数的调优方案
"""
#导包
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

#加载数据
iris_data=load_iris()

#数据预处理（划分数据集）
#参1：数据特征，参2：数据标签，参3：测试集占比，参4：随机数种子
x_train,x_test,y_train,y_test=train_test_split(iris_data.data,iris_data.target,test_size=0.25,random_state=37)

#特征工程
#特征预处理（标准化）
#创建标准化对象
transfer=StandardScaler()
#对训练特征和测试特征进行标准化
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

#模型训练
#创建分类器
estimator=KNeighborsClassifier()    #不指定超参数，后续进行网格搜索
#定义字典，记录超参数可能会出现的情况
param_proba={'n_neighbors':[i for i in range(1,11)]}
#创建GridSearchCV对象——>寻找最优超参数，使用网格搜索+交叉验证方式
#参数1：要计算最优超参的模型对象，参数2：该模型可能的超参数值,参数3：交叉验证次数
estimator=GridSearchCV(estimator,param_grid=param_proba,cv=4)
#实际训练
estimator.fit(x_train,y_train)

#打印最优超参数、最优评分、最优估计器对象
print(f'最优超参数：{estimator.best_params_}')
print(f'最优评分：{estimator.best_score_}')
print(f'最优估计器：{estimator.best_estimator_}')
print(f'具体的交叉验证结果：{estimator.cv_results_}')

#模型评估
#获取最优模型对象
estimator=estimator.best_estimator_
#模型训练
# estimator=estimator.fit(x_train,y_train)  #如果是通过最优模型函数得到的，可以不训练
#模型预测
y_pred=estimator.predict(x_test)
print(f'准确率为{accuracy_score(y_test,y_pred)*100}%')
#GridSearchCV 有一个参数：refit=True，它默认就是 True，表示找到最优超参数后，自动使用全部训练数据重新拟合最优模型。estimator = grid.best_estimator_得到的已经是拟合好的模型

