# -*- encoding: utf-8 -*-
"""
@File    : 01_特征选择-filter.py
@Time    : 2022/4/21 0021 13:43
@Author  : L
@Software: PyCharm
"""

from sklearn.feature_selection import VarianceThreshold

# 移除低方差的特征: 把第一列特征给干掉了
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
print(X)
# 移除方差小于 0.16的列
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_v = sel.fit_transform(X)
print(X_v)

# 卡方(Chi2)检验

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

iris = load_iris()
X, y = iris.data, iris.target
print(X.shape)
selector = SelectKBest(chi2, k=2)
X_new = selector.fit_transform(X, y)
print(X_new.shape)

# Pearson相关系数 (Pearson Correlation)
import numpy as np
from scipy.stats import pearsonr

np.random.seed(0)
size = 300
x = np.random.normal(0, 1, size)
print("Lower noise：", pearsonr(x, x + np.random.normal(0, 1, size)))
print("Higher noise：", pearsonr(x, x + np.random.normal(0, 10, size)))

from sklearn.feature_selection import SelectKBest
from sklearn import datasets

iris = datasets.load_iris()
# 构建相关系数模型
model = SelectKBest(lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T[0], k=2)
model.fit_transform(iris.data, iris.target)  # 对模型将数据传入
print('相关系数：', model.scores_)  # 返回所有变量X与Y的相关系数值
print('P值：', model.pvalues_)  # 返回所有变量X的P值
# 打印传入数据的话会返回k=2所选择的两个变量的数据的值
print('所选变量的数值为：\n', model.fit_transform(iris.data, iris.target))

# 互信息
#
# import numpy as np
# from sklearn.feature_selection import SelectKBest
# from minepy import MINE
# from sklearn import datasets
#
# iris = datasets.load_iris()
#
#
# def mic(x, y):
#     m = MINE()
#     m.compute_score(x, y)
#     return (m.mic(), 0.5)
#
#
# # 构建互信息数模型
# model = SelectKBest(lambda X, Y: np.array(list(map(lambda x: mic(x, Y), X.T))).T[0], k=2)
# model.fit_transform(iris.data, iris.target)  # 对模型将数据传入
# print('互信息系数：', model.scores_)  # 返回所有变量X与Y的相关系数值
# print('P值：', model.pvalues_)  # 返回所有变量X的P值
# # 打印传入数据的话会返回k=2所选择的两个变量的数据的值


# 基于模型的特征排序(Model based ranking)
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load boston housing dataset as an example
boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
# 单独采用每个特征进行建模，并进行交叉验证
for i in range(X.shape[1]):
    score = cross_val_score(rf, X[:, i:i + 1], Y, scoring="r2",  # 注意X[:, i]和X[:, i:i+1]的区别
                            cv=ShuffleSplit(n_splits=10, test_size=0.3))
    scores.append((format(np.mean(score), '.3f'), names[i]))
print(sorted(scores, reverse=True))



