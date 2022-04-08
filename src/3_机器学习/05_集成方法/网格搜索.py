# -*- encoding: utf-8 -*-
"""
@File    : 网格搜索.py
@Time    : 2022/4/8 0008 14:06
@Author  : L
@Software: PyCharm
"""
"""
    Simple Grid Search：简单的网格搜索
"""
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
print("Size of training set:{} size of testing set:{}".format(X_train.shape[0], X_test.shape[0]))

####   grid search start
best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)  # 对于每种参数可能的组合，进行一次训练；
        svm.fit(X_train, y_train)
        score = svm.score(X_test, y_test)
        if score > best_score:  # 找到表现最好的参数
            best_score = score
            best_parameters = {'gamma': gamma, 'C': C}
####   grid search end

print("Best score:{:.2f}".format(best_score))
print("Best parameters:{}".format(best_parameters))

"""
存在的问题：
原始数据集划分成训练集和测试集以后，其中测试集除了用作调整参数，也用来测量模型的好坏；这样做导致最终的评分结果比实际效果要好。
（因为测试集在调参过程中，送到了模型里，而我们的目的是将训练模型应用在unseen data上）；

解决方法：
对训练集再进行一次划分，分成训练集和验证集，这样划分的结果就是：原始数据划分为3份，
分别为：训练集、验证集和测试集；其中训练集用来模型训练，验证集用来调整参数，而测试集用来衡量模型表现好坏。
"""

X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=1)
print(
    "Size of training set:{} size of validation set:{} size of teseting set:{}".format(X_train.shape[0], X_val.shape[0],
                                                                                       X_test.shape[0]))

best_score = 0.0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_val, y_val)
        if score > best_score:
            best_score = score
            best_parameters = {'gamma': gamma, 'C': C}
svm = SVC(**best_parameters)  # 使用最佳参数，构建新的模型
svm.fit(X_trainval, y_trainval)  # 使用训练集和验证集进行训练，more data always results in good performance.
test_score = svm.score(X_test, y_test)  # evaluation模型评估
print("Best score on validation set:{:.2f}".format(best_score))
print("Best parameters:{}".format(best_parameters))
print("Best score on test set:{:.2f}".format(test_score))

"""
    然而，这种间的的grid search方法，其最终的表现好坏与初始数据的划分结果有很大的关系，为了处理这种情况，我们采用交叉验证的方式来减少偶然性。
"""
from sklearn.model_selection import cross_val_score

best_score = 0.0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)  # 5折交叉验证
        score = scores.mean()  # 取平均数
        if score > best_score:
            best_score = score
            best_parameters = {"gamma": gamma, "C": C}
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)
print("Best score on validation set:{:.2f}".format(best_score))
print("Best parameters:{}".format(best_parameters))
print("Score on testing set:{:.2f}".format(test_score))
