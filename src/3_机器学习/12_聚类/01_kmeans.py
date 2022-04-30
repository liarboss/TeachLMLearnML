#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
# @Time : 2022/4/30 10:48 
# @Author : cl
# @File : 01_kmeans.py 
# @Software: PyCharm
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()
X = iris.data[:, :4]  # #表示我们取特征空间中的4个维度
print(X.shape)

# 绘制数据分布图
# plt.figure(figsize=(15,8),dpi=80)
plt.scatter(X[:, 0], X[:, 1], c="red", marker='*', label='see')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()

estimator = KMeans(n_clusters=3)  # 构造聚类器
estimator.fit(X)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签
# 绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.figure(figsize=(15, 8), dpi=80)
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()