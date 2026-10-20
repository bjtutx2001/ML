"""
KNN算法介绍（K Nearest Neighbors），K近邻算法
    原理：基于欧氏距离(默认)计算测试集与每个训练集之间的距离，根据距离生序排序，找到距离最近的K个样本
        基于K个样本投票，票数最多的作为最终测试结果——>分类问题
        基于K个样本计算平均值，作为最终预测结果——>回归问题
    实现思路：
        1、分类问题：适用于有特性，有标签，标签不连续（离散）
        2、回归问题：适用于有特性，有标签，标签连续
    KNN回归思路：
        1、计算距离、升序
        2、找到最近的N个训练样本
        3、计算N个样本距离的平均值
        4、以评价值作为预测结果
    KNN回归代码实现思路：
        1、导包
        2、准备数据集（测试集、训练集）
        3、创建KNN分类模型对象
        4、模型训练
        5、模型预测
    总结：
        K值过小，容易受到异常值的影响，且会收到模型学到大量的脏数据，导致出现：过拟合
        K值过大，容易使模型过于简单，导致欠拟合

"""
#导包
from sklearn.neighbors import KNeighborsRegressor

#准备数据集
x_train=[[1,2,3],[2,3,5],[10,2,7],[6,5,8]]
y_train=[0.1,0.2,0.3,0.4]
x_test=[[5,3,9]]
#创建KNN分类模型
#estimator:估计器，模型对象，也可以用其他变量名
estimator=KNeighborsRegressor(n_neighbors=3)
#训练模型
estimator.fit(x_train,y_train)
#预测模型
y_predict=estimator.predict(x_test)
print(f"预测结果为：{y_predict}")

