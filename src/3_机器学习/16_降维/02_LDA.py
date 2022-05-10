# -*- encoding: utf-8 -*-
"""
@File    : 02_LDA.py
@Time    : 2022/5/10 0010 13:33
@Author  : L
@Software: PyCharm
"""
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 定义可视化函数用于结果展示
# 可视化函数
def plot_decision_regions(x, y, classifier, resolution=0.02):
    markers = ['s', 'x', 'o', '^', 'v']
    colors = ['r', 'g', 'b', 'gray', 'cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)

    for idx, cc in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cc, 0],
                    y=x[y == cc, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cc)

# 拟合数据
#数据集来源
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)

#切割数据集
#x数据
#y标签
x, y = data.iloc[:, 1:].values, data.iloc[:, 0].values

#按照8:2比例划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

#标准化单位方差
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)

lda = LDA(n_components=2)
lr = LogisticRegression()

#训练
x_train_lda = lda.fit_transform(x_train_std, y_train)
#测试
x_test_lda = lda.fit_transform(x_test_std, y_test)
#拟合
lr.fit(x_train_lda, y_train)


# 结果展示
# 画图高宽，像素
plt.figure(figsize=(6, 7), dpi=100)
plot_decision_regions(x_train_lda, y_train, classifier=lr)
plt.show()

