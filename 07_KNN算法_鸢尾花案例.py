"""
机器学习的研发流程：
    1、加载数据
    2、数据预处理
    3、特征工程
    4、模型训练
    5、模型评估
    6、模型预测
演示鸢尾花案例

"""
#导包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris                  #鸢尾花数据
from sklearn.neighbors import KNeighborsClassifier       #KNN分类器
from sklearn.model_selection import train_test_split    #分割数据
from sklearn.preprocessing import StandardScaler        #数据标准化
from sklearn.metrics import accuracy_score              #数据评估

# 设置中文字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 解决负号显示异常的问题
plt.rcParams['axes.unicode_minus'] = False

#1、定义函数，加载鸢尾花数据，并查看数据详情
def load_data():
    #加载数据
    iris_data=load_iris()
    #查看数据集
    # print(iris_data.data)
    # print(type(iris_data.data))     #不是字典，但可以看成字典
    # print(iris_data.keys())         #dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
    # print(iris_data.data)           #具体数据
    # print(iris_data.target)         #鸢尾花类别
    # print(iris_data.target_names)         #鸢尾花分类名字

#2、鸢尾花数据可视化
def show_data():
    iris_data=load_iris()
    #把鸢尾花数据集封装成dataframe
    iris_df=pd.DataFrame(iris_data.data,columns=iris_data.feature_names)
    iris_df['target']=iris_data.target
    # print(iris_df)
    #绘制散点图，添加拟合回归线,lmplot可以设置hue参数绘制多个拟合回归线
    #参1：数据，参2：x轴，参3：y轴，参4：分类依据，参5：是否绘制拟合回归线，参6：线条参数
    sns.lmplot(data=iris_df,x='sepal length (cm)',y='sepal width (cm)',hue='target',fit_reg=True,line_kws={'color':'red'})
    plt.title('鸢尾花散点图')
    plt.tight_layout()      #调整绘图布局
    plt.show()

#3、切分训练集和测试集
def cut_data():
    #加载数据
    iris_data=load_iris()
    #切割测试集与训练集
    #参1：数据特征，参2；数据标签，参3：测试集比例，参4：随机数种子，设置了每次生成所选择的数据会一样
    #返回结果训练特性，训练标签，测试特征，测试标签
    x_train,x_text,y_train,y_text=train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=37)
    print(f'训练集的数据特征为:{x_train},个数为{len(x_train)}')

#定义函数，实现完整鸢尾花案例
def dm04_iris_evaluate():
    #1、加载数据，
    iris_data=load_iris()
    # 2、数据预处理，将数据按照8:2划分为测试集和训练集
    x_train,x_test,y_train,y_test=train_test_split(iris_data.data,iris_data.target,test_size=0.3,random_state=37)
    # 3、特征工程
    # 3.1、特征提取,此处源数据就是四种特征，无需提取
    # 3.2、特征预处理，因为源数据的四列特征差距不大，实际开发面对这种情况无需预处理，但是加入预处理会使代码更完善，此处采用标准化
        #创建标准化对象
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)     #fit函数会得到数据集的标准化信息（归一化信息），fit_trans则是直接采用这套信息进行标准化
    x_test=transfer.transform(x_test)           #trans（）函数会直接沿用之前的标准化信息进行标准化，如果这里使用fit_trans()函数，集的均值和方差将会和训练集不一致，二者沿用不一样的标准化规则，会导致模型训练出错

    # 3.3、特征降维，略
    # 3.4、特征选择，略
    # 3.5、特征组合，略

    # 4、模型训练
    estimator=KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)      #传入训练集的特征和标签
    # 6、模型预测
    y_predict=estimator.predict(x_test) #预测测试集标签
    print(f'预测结果为:{y_predict}')
    # 5.1、使用新数据集预测
    my_data=[[4.7,2.4,6.8,2.4]]
    my_pre=estimator.predict(my_data)
    print(my_pre)
    print(f'my_data各个分类的预测概率为:{estimator.predict_proba(my_data)}')
    # 6、模型评估(正确率)
    # 6.1、score基于测试集的特征和测试集的标签，参1：测试集特征，参2：测试集标签
    print(f'准确率为：{estimator.score(x_test,y_test)}')
    # 6.1、score基于测试集的标签和预测结果参1：测试集标签，参2：预测结果
    print(f'准确率为：{accuracy_score(y_test,y_predict)}')       #两种评估方法得到的结果应该是一样的

#测试
if __name__=="__main__":
    # load_data()
    # show_data()
    # cut_data()
    dm04_iris_evaluate()