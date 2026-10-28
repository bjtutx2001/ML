"""
逻辑回归
    介绍：属于有监督学习，有特征，有标签，标签离散，主要用于二分类
    原理：把线性回归后的数据，通过sigmoid函数激活，映射到[0,1]区间，基于自定义阈值结合概率进行二分类
    损失函数：极大似然估计的负数形式
    正类：一般情况下默认样本数少的为正类
    评估方法：
        1、精确率：预测为正类的样本，实际为正类的比例
        2、召回率：实际为正类的样本被预测为正类的比例
        3、F1——score：精确率和召回率的调和平均

    混淆矩阵：原来描述实际值和预测值之间的关系
        形式：
                                            预测标签（正类）        预测标签（反类）
                真实标签（正类）          TP  True Positive       FN  False Negative
                真实标签（反类）          FP  False Negative      TN  True Positive

        精确率=TP/(TP+FP)
        召回率=TP/(TP+FN)
        F1-Score=2(精确率*召回率)/(精确率+召回率)

    ROC曲线和AUC
        ROC：
            描述：ROC曲线的横坐标为假正例率（FP/(FP+TN)），纵坐标为真正例率(召回率)
            如何生成：逻辑回归中，二分类的判断依据为自定义阈值，假设一个样本的激活值为0.8，若阈值为0.9（大1，小0），则样本被预测为0，通过改变阈值，会导致样本的预测准确率
                    ROC曲线就是在多次改变阈值的情况下绘制不同阈值的假正例和真正例值（一个阈值为一个点，后续连线）
            特性：
                1、阈值越高，模型越谨慎地预测正例，通常 TPR 和 FPR 都较低。
                2、阈值越低，模型越容易预测正例，通常 TPR 和 FPR 都较高。
        AUC：
            描述：就是ROC曲线下的面积，取值在0——1之间

"""
#导包
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split    #数据集划分
from sklearn.linear_model import LogisticRegression     #逻辑回归
from sklearn.preprocessing import StandardScaler        #标准化
from sklearn.metrics import confusion_matrix            #混淆矩阵

#准备数据
data=pd.read_csv('./data/breast-cancer-wisconsin.csv')  #第一列为标签

#缺失值处理
data.replace('?',np.nan,inplace=True)
data.dropna(axis=0,inplace=True)
# data.info()
#分离特征
x = data.iloc[:,1:-1]   #获取除了第一类和最后一列
y = data.iloc[:,-1]     #获取最后一列

#特征工程
# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=37)
# 特征标准化
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

#模型训练
estimator=LogisticRegression()
estimator.fit(x_train,y_train)

#模型预测
y_pred=estimator.predict(x_test)

#模型评估
print(f'预测的正确率为：{estimator.score(x_test,y_test)}')
# print(type(y_pred))
# print(y_pred)
# y_test=y_test.values    #将series转换为ndarray对象
# print(type(y_test))
# print(y_test)

#创建混淆矩阵，
label=[4,2]
CM_1=confusion_matrix(y_test,y_pred,labels=label)    #参1：真实值，参2：预测值，参3：矩阵的显示顺序第一行与第一列为4，第二行与第二列为2
# print(CM_1)
# print(type(CM_1))

#封装成Df对象，添加索引和标签
CM_1=pd.DataFrame(CM_1,index=['真例(4)','假例(2)'],columns=['真例(4)','假例(2)'])
print(CM_1)

#计算精确率，召回率，F1_Score
print(f'预测结果的精确率为{precision_score(y_test,y_pred,pos_label=4,average='binary')}')  #参1：真实值，参2：预测值，参3：正例，参4：计算方式，默认为二分类
print(f'预测结果的召回率为{recall_score(y_test,y_pred,pos_label=4,average='binary')}')  #参1：真实值，参2：预测值，参3：正例，参4：计算方式，默认为二分类
print(f'预测结果的F1值为{f1_score(y_test,y_pred,pos_label=4,average='binary')}')  #参1：真实值，参2：预测值，参3：正例，参4：计算方式，默认为二分类
