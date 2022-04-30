#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
# @Time : 2022/4/30 10:28 
# @Author : cl
# @File : 05_DBSCAN.py 
# @Software: PyCharm
"""

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import mglearn

X,y=make_blobs(random_state=0,n_samples=12)
dbscan=DBSCAN()
clusters=dbscan.fit_predict(X)
# 都被标记为噪声
print('Cluster memberships:\n{}'.format(clusters))
mglearn.plots.plot_dbscan()

plt.show()