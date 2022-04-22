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
