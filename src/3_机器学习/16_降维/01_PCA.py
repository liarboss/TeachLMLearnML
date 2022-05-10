# -*- encoding: utf-8 -*-
"""
@File    : 01_PCA.py
@Time    : 2022/5/10 0010 9:28
@Author  : L
@Software: PyCharm
"""
from sklearn.decomposition import PCA
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)

# 结果 [0.99244289 0.00755711]


pca = PCA(n_components='mle')
pca.fit(X)
print(pca.explained_variance_ratio_)

# 结果 0.99244289

"""
n_components：表示保留的特征数，默认为1，如果设置成‘mle’,那么会自动确定保留的特征数。
n_components 设置成 mle时，自动确定保留的特征数，此时特征数为1。
explained_variance_ratio_：返回所保留的n个成分各自的方差百分比,这里可以理解为单个变量方差贡献率。

可以看到第一个特征的单个变量方差贡献率已经到达0.99,意味着几乎保留了所有的信息，所以只保留一个特征即可。

n_features_：训练数据中的特征数量
n_samples_：训练数据中的样本数
"""

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
print(pca.n_features_)
print(pca.n_samples_)

# 结果 ：2  6

"""
方法说明：
方法	说明
fit(X)	用数据X来训练PCA模型。
transform(X)	将数据X转换成降维后的数据，当模型训练好后，对于新输入的数据，也可以用transform方法来降维。
fit_transform(X)	用X来训练PCA模型，同时返回降维后的数据。
inverse_transform(newData)	newData 为降维后的数据。将降维后的数据转换成原始数据，但可能不会完全一样，会有些许差别。
"""
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components='mle')
result=pca.fit_transform(X)
print(result)


