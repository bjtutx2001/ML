"""
数据标准化就是把数据映射到均值为0，方差为1的标准正态分布数据
计算方式：
    x'=(x-mean)/σ
"""
#导包
from sklearn.preprocessing import StandardScaler

#准备数据
x_train=[[1,2,3],[4,5,6],[7,8,9]]

#创建标准化对象
transfer=StandardScaler()

#标准化
x_train_new=transfer.fit_transform(x_train)

#打印数据
print(f"标准化的结果为:{x_train_new}")

#查看原数据的均值和方差
#这里想使用mean_()方法查看数据的均值必须要对源数据调用标准化的fit以及衍生方法才行，否则会报错
#fit在KNN中表示拟合，在这里表示计算均值和方差，这两个函数虽然不一样，但是遵循sklearn统一的接口设计，核心都是根据数据学习或确定后续操作所需的信息
print(f"源数据均值为:{transfer.mean_}")
print(f"源数据方差为:{transfer.var_}")