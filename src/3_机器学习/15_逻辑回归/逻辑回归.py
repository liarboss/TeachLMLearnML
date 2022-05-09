# -*- encoding: utf-8 -*-
"""
@File    : 逻辑回归.py
@Time    : 2022/5/9 0009 10:20
@Author  : L
@Software: PyCharm
"""

import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# 1.加载数据
iris = datasets.load_iris()
X = iris.data[:, :2]  # 使用前两个特征
Y = iris.target

# 2.拆分测试集、训练集。
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# 设置随机数种子，以便比较结果。

# 3.标准化特征值
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
num = 0

# 4. 训练逻辑回归模型

logreg = linear_model.LogisticRegression(penalty='l2', C=1e5, multi_class="ovr")
logreg.fit(X_train_std, Y_train)

# 5. 预测
predic = logreg.predict(X_test_std)
# p = np.mean(predic == Y_test)
for i in X_test_std:
    print(i, Y_test[num], predic[num])
    num += 1
prepro = logreg.predict_proba(X_test_std)

acc = logreg.score(X_test_std, Y_test)
print(prepro)

scores = cross_val_score(logreg, X_train_std, Y_train)
print(np.mean(scores))
