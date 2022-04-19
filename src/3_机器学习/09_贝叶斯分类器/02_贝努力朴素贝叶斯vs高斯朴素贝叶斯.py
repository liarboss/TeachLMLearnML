# -*- encoding: utf-8 -*-
"""
@File    : 02_贝努力朴素贝叶斯vs高斯朴素贝叶斯.py
@Time    : 2022/4/19 0019 16:15
@Author  : L
@Software: PyCharm
"""
# 导入数据集
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np

# 这里构建了一个500个样本，5类的数据
from sklearn.naive_bayes import BernoulliNB

x, y = make_blobs(n_samples=500, centers=5, random_state=8)
print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=8)
nb = BernoulliNB()
nb.fit(x_train, y_train)
print(nb.score(x_test, y_test))

# 画图
import matplotlib.pyplot as plt

# 设置横轴与纵轴最大值
x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
# 用不同的北背景表示分类
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
z = nb.predict(np.c_[(xx.ravel(), yy.ravel())]).reshape(xx.shape)
plt.pcolormesh(xx, yy, z, cmap=plt.cm.Pastel1)
# 将测试集与训练集用散点图表示
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.cool, edgecolor='k')
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.cool, marker="*", edgecolor="k")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("BERNOULLINB")
plt.show()

# 高斯朴素贝叶斯
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train, y_train)
gnb.score(x_test, y_test)
# 画图
z = gnb.predict(np.c_[(xx.ravel(), yy.ravel())]).reshape(xx.shape)
plt.pcolormesh(xx, yy, z, cmap=plt.cm.Pastel1)
# 将测试集与训练集用散点图表示
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.cool, edgecolor='k')
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.cool, marker="*", edgecolor="k")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("gaussianNB")
plt.show()
