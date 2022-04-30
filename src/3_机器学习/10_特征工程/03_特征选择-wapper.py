# -*- encoding: utf-8 -*-
"""
@File    : 03_特征选择-wapper.py
@Time    : 2022/4/22 0022 15:15
@Author  : L
@Software: PyCharm
"""

# Wrapper - 递归特征消除

import numpy as np
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
Y = iris.target
# FECV()利用交叉验证来选择，不过这里的交叉验证的数据集切割对象不再是 行数据（样本），
# 而是列数据（特征），但是计算量会大，cv默认是3
# 构建逻辑回归的递归消除模型
model = RFE(estimator=LogisticRegression(), n_features_to_select=2)  # 构建逻辑回归的递归消除模型
X_new = model.fit_transform(iris.data, iris.target)  # 传入数据
print(X[:1])  # [[5.1 3.5 1.4 0.2]]
print(X_new[:1])  # [[3.5 0.2]]


# SelectFromModel
#（1）基于L1、L2惩罚项的特征选择法
# 使用带惩罚项的基模型，除了筛选出特征外，同时也进行了降维。使用feature_selection库的SelectFromModel类结合带L1惩罚项的逻辑回归模型，
# 来选择特征的代码如下：

# import  numpy as np
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LogisticRegression
# from sklearn import datasets
# iris = datasets.load_iris()
# X = iris.data
# Y = iris.target
# # 带L1惩罚项的逻辑回归作为基模型的特征选择
# model = SelectFromModel(LogisticRegression(penalty="l1", C=0.1), max_features=2)
# model.fit_transform(iris.data, iris.target) # 传入数据
# # 返回模型选择的变量的数据内容
# X_new = model.fit_transform(X, Y)
# print(X[:1])# [[5.1 3.5 1.4 0.2]]
# print(X_new[:1])# [[3.5 1.4]]


# 基于树的特征选择

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
iris = load_iris()
X, Y = iris.data, iris.target
print(X.shape)
clf = ExtraTreesClassifier()
clf = clf.fit(X, Y)
print(clf.feature_importances_)# [0.14419482 0.05850172 0.41185762 0.38544583]
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
print(X_new.shape)

# import numpy as np
# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn import datasets
#
# iris = datasets.load_iris()
#
# # 带L1惩罚项的逻辑回归作为基模型的特征选择
# model = SelectFromModel(GradientBoostingClassifier())
# model.fit_transform(iris.data, iris.target)  # 传入数据
# print(model.fit_transform(iris.data, iris.target))  # 返回模型选择的变量的数据内容

