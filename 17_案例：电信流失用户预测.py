"""
案例；典型用户流失数据预测

"""
#导包
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#定义函数，数据预处理
def dm01_data_preprocessing():
    data=pd.read_csv('./data/churn.csv',sep=',',encoding='utf-8')
    # print(data.head())
    # data.info()
    #机器学习的特征数据必须是数值类型，这里由于有两列为字符串类型，进行one-hot编码（独热编码）
    #独热编码：将类别型数据转换成数值型数据，类别型数据必须是没有顺序（类别结果之展示差异，无好坏之分，如男女，颜色，而成绩优良就不行）
    data=pd.get_dummies(data,columns=['Churn','gender'])
    # print(data.info())
    data=data.drop(columns=['Churn_No','gender_Male'])
    #指定Churn_No为标签列
    data=data.rename(columns={'Churn_Yes':'Flag'})
    print(data.info())

#定义函数，数据可视化
def dm02_data_visualization():
    data = pd.read_csv('./data/churn.csv',sep=',',encoding='utf-8')
    data=pd.get_dummies(data,columns=['Churn','gender'])
    data = data.drop(columns=['Churn_No', 'gender_Male'])
    data = data.rename(columns={'Churn_Yes': 'Flag'})
    #查看会员的人数情况
    sns.countplot(data=data,x='Contract_Month',hue='Flag')     #参1：数据源，参2：指定x轴数据
    plt.show()
    # print(data.columns)
    df=data['Contract_Month'].value_counts()
    print(df)


#定义函数，训练模型，预测是否流失
def dm03_model():
    #数据预处理
    data = pd.read_csv('./data/churn.csv',sep=',',encoding='utf-8')
    data=pd.get_dummies(data,columns=['Churn','gender'])
    data=data.drop(columns=['Churn_No','gender_Male'])
    data=data.rename(columns={'Churn_Yes':'Flag'})
    """
    列名
    'Partner_att', 'Dependents_att', 'landline', 'internet_att',
       'internet_other', 'StreamingTV', 'StreamingMovies', 'Contract_Month',
       'Contract_1YR', 'PaymentBank', 'PaymentCreditcard', 'PaymentElectronic',
       'MonthlyCharges', 'TotalCharges', 'Flag', 'gender_Female'
    """
    #特征提取
    x=data[['Partner_att','Dependents_att','landline','internet_att',
       'internet_other', 'StreamingTV', 'StreamingMovies', 'Contract_Month',
       'Contract_1YR', 'PaymentBank']]
    y=data['Flag']  #False不流失
    #划分数据集
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=37)
    print(data.head())

    #标准化
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    #模型训练
    estimator=LogisticRegression()
    estimator.fit(x_train,y_train)

    #模型预测
    y_pred=estimator.predict(x_test)

    #模型评估
    print(f"模型预测的正确率为：{estimator.score(x_test,y_test)}")
    print(f"模型预测的正确率为：{accuracy_score(y_test,y_pred)}")
    print(f"模型精确率为：{precision_score(y_test,y_pred)}")
    print(f"模型召回率为：{recall_score(y_test,y_pred)}")
    print(f"模型F1值为：{f1_score(y_test,y_pred)}")

    print(f"分类评估报告：{classification_report(y_test,y_pred)}")




#测试
if __name__ == '__main__':
    # dm01_data_preprocessing()
    # dm02_data_visualization()
    dm03_model()