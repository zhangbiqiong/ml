from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

'''载入数据'''
# X,y = datasets.load_iris(return_X_y=True)

X=[
    [32.0,33.0],
    [36.0,44.0],
    [16.0,4.0],
    [16.0,44.0],
    [3.0,14.0],
    [22.0,59.0],
    [36.0,54.0],
    [56.0,4.0],
    [56.0,14.0],
    [9.0,14.0],

    [72.0,73.0],
    [76.0,74.0],
    [86.0,84.0],
    [86.0,64.0],
    [86.0,74.0],
    [92.0,73.0],
    [60.0,74.0],
    [86.0,64.0],
    [96.0,64.0],
    [86.0,60.0],

]

y=[0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1]


'''分割训练集与验证集，这里采用分层抽样的方法控制类别的先验概率'''
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y)

'''初始化高斯朴素贝叶斯分类器'''
clf = GaussianNB()

'''训练分类器'''
clf = clf.fit(X_train,y_train)

print(clf.predict([[60.0,90.0]]))

'''打印分类器在验证集上的混淆矩阵'''
print('混淆矩阵：')
print(confusion_matrix(y_test,clf.predict(X_test)))

'''打印测试集上的正确率'''
print('测试集正确率：'+str(clf.score(X_test,y_test)))
'''打印分类器训练后的各返回项'''
# print('类别的先验分布：',clf.class_prior_)
# print('各类别样本数量：',clf.class_count_)
# print('各类别对应各连续属性的正态分布的均值：','\n',clf.theta_)
# print('各类别对应各连续属性的正态分布的方差：','\n',clf.sigma_)
