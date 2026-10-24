"""
案例：演示采用KNN算法实现手写数字识别

数据介绍：每张图片有28*28个像素点构成，像素的取值代表数字的轮廓
"""

#导包
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
from collections import Counter

#扩展：忽略警告,警告原因：训练时用的df带有列名，但是预测用的numpy数组没有列名字
import warnings
warnings.filterwarnings('ignore',module='sklearn')

#定义函数，接受用户传入的索引，展示该索引对应的图片
def show_digit(idx):
    #读取数据
    df=pd.read_csv('./data/手写数字识别.csv')
    # print(df)
    # print(df.index)
    # print(df.columns)

    #判断传入的索引是否越界
    if idx < 0 or idx >= len(df):
        print(f'索引越界')
        return

    #获取正常的数据
    #获取标签
    y=df.iloc[:,0]
    #获取特征
    x=df.iloc[:,1:]

    #查看用户传入的索引对应的图片是数字几
    print(f'图片{idx}对应的数字是{y.iloc[idx]}')
    print(f'所有数字的统计结果为{Counter(y)}')

    #重新编排图片成原始形状
    # print(x.iloc[idx,:].shape)
    x=x.iloc[idx,:].values.reshape(28,28)
    # print(x.shape)
    # print(x)

    #绘制灰度图
    plt.imshow(x,cmap='gray')
    #关闭坐标显示
    plt.axis('off')
    plt.show()

#定义函数，训练模型
def train_model():
    #加载数据
    df=pd.read_csv('./data/手写数字识别.csv')
    #数据预处理，拆出特征数据和标签数据
    x=df.iloc[:,1:]
    y=df.iloc[:,0]
    #查看所有标签分布情况
    print(f'所有数字分布情况：{Counter(y)}')

    #这里由于像素的特殊情况，可以手动进行归一化，无需使用归一化工具
    x = x / 255

    #数据预处理
    result=train_test_split(x,y,test_size=0.2,random_state=37,stratify=y)
    x_train,x_test,y_train,y_test=result

    #模型训练
    estimator=KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)

    #保存模型
    joblib.dump(estimator,'./model/手写数字识别.pkl')

    #模型评估
    print(f'模型准确率为{estimator.score(x_test,y_test)}')
    print(f'模型准确率为{accuracy_score(y_test,estimator.predict(x_test))}')

#定义函数，测试模型
# def test_model():   #函数名字以test开头，python会自动执行pytest模块进行测试
def use_model():
    #读取图片
    x=plt.imread('./data/demo.png')
    # plt.imshow(x,cmap='gray')
    # plt.axis('off')
    # plt.show()

    #转换图片格式：28*28——>1*748；并归一化
    #imread读取png图片会自动归一化像素，因此预测时除以255会导致预测失败
    x=x.reshape(1,-1)   #reshape(1,-1)语法糖，-1表示自动获取列数

    #加载模型
    estimator=joblib.load('./model/手写数字识别.pkl')

    #模型预测
    y_pre=estimator.predict(x)

    print(f'预测结果为{y_pre}')


if __name__=='__main__':
    # show_digit(37)
    # train_model()
    use_model()