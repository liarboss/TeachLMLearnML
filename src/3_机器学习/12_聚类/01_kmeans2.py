#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
# @Time : 2022/4/30 11:09 
# @Author : cl
# @File : 01_kmeans2.py 
# @Software: PyCharm
"""

import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
# X为样本特征，Y为样本簇类别， 共1000个样本，
# 每个样本4个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1],[2,2]， 簇方差分别为[0.4, 0.2, 0.2]
X, y = make_blobs(n_samples=1000, n_features=2,
centers=[[-1,-1], [0,0], [1,1], [2,2]],
cluster_std=[0.4, 0.2, 0.2, 0.2],
random_state =9)
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()


from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
#用Calinski-Harabasz Index评估二分类的聚类分数
print(metrics.calinski_harabasz_score(X, y_pred))
#Calinski-Harabasz Index对应的方法是metrics.calinski_harabaz_score


from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
print(metrics.calinski_harabasz_score(X, y_pred))
#用Calinski-Harabasz Index评估三分类的聚类分数


from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
print(metrics.calinski_harabasz_score(X, y_pred))
#用Calinski-Harabasz Index评估四分类的聚类分数