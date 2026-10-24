"""
    特征工程的目的和步骤：
        目的：
            利用专业的背景知识和技巧处理数据，用于提升模型的性能
        步骤：
            1、特征提取
            2、特征预处理
            3、特征降维
            4、特征选择
            5、特征组合
    归一化是特征工程的常见技术手段
    归一化介绍：
        防止因为量纲（单位）问题，导致特征列的方差较大，影响模型的最终结果
        公式：
            x'=（当前值-最小值）/（最大值-最小值）
        弊端：容易受到最大值和最小值的影响，一般用于处理小数据集
"""
#导包
from sklearn.preprocessing import MinMaxScaler

#准备数据
x_train=[[25,31,43,5],[44,15,22,33],[9,26,42,51],[14,5,7,21]]

#创建归一化对象
#scaler为定标器的意思，也可以使用其他变量，例如transfer
scaler = MinMaxScaler(feature_range=(0,1))  #feature_range设定归一化区间，默认为（0,1）

#对数据集进行归一化操作
x_train_new=scaler.fit_transform(x_train)

#打印处理后的数据
print(f"归一化后的数据为：{x_train_new}")