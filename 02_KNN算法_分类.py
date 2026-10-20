"""
KNN算法介绍（K Nearest Neighbors），K近邻算法
    原理：基于欧氏距离(默认)计算测试集与每个训练集之间的距离，根据距离生序排序，找到距离最近的K个样本
        基于K个样本投票，票数最多的作为最终测试结果——>分类问题
        基于K个样本计算平均值，作为最终预测结果——>回归问题
    实现思路：
        1、分类问题：适用于有特性，有标签，标签不连续（离散）
        2、回归问题：适用于有特性，有标签，标签连续
    KNN分类思路：
        1、计算测试集与训练集的思路，升序
        2、找到距离最近的K个训练样本
        3、根据K个样本的标签对测试集的标签进行投票
        4、取票数最多的作为结果
    KNN分类代码实现思路：
        1、导包
        2、准备数据集（测试集、训练集）
        3、创建KNN分类模型对象
        4、模型训练
        5、模型预测

"""
#导包
from sklearn.neighbors import KNeighborsClassifier

#准备数据集
x_train=[[1],[2],[3],[4],[5],[6],[7],[8]]
y_train=[0,0,0,0,1,1,1,1]
x_test=[[5],[1]]
#创建KNN分类模型
#estimator:估计器，模型对象，也可以用其他变量名
estimator=KNeighborsClassifier(n_neighbors=3)
#训练模型
estimator.fit(x_train,y_train)
#预测模型
y_predict=estimator.predict(x_test)
print(f"预测结果为：{y_predict}")

