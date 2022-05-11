# -*- encoding: utf-8 -*-
"""
@File    : 03_SVD降维.py
@Time    : 2022/5/11 0011 13:37
@Author  : L
@Software: PyCharm
"""
from sklearn.datasets import load_digits
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

X, y = load_digits(return_X_y=True)
X.shape
y.shape
rfc = RandomForestClassifier()
# 交叉验证10次
scroe = cross_val_score(rfc, X, y, cv=10)
print(scroe.mean())
# 降至16个特征维
tsvd = TruncatedSVD(n_components=16)
X_tsvd = tsvd.fit_transform(X)
scroe = cross_val_score(rfc, X_tsvd, y, cv=10)
print(scroe.mean())
# 降至8个特征维
tsvd = TruncatedSVD(n_components=8)
X_tsvd = tsvd.fit_transform(X)
scroe = cross_val_score(rfc, X_tsvd, y, cv=10)
print(scroe.mean())
