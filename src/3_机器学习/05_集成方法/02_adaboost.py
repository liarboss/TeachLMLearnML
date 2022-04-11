# -*- encoding: utf-8 -*-
"""
@File    : 02_adaboost.py
@Time    : 2022/4/11 0011 10:43
@Author  : L
@Software: PyCharm
"""

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# 加载数据集，并按照8:2的比例分割成训练集和测试集
wine = load_wine()
print(f"所有特征：{wine.feature_names}")
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# 使用单一决策树建模
base_model = DecisionTreeClassifier(max_depth=1, criterion='gini', random_state=1).fit(X_train, y_train)
y_pred = base_model.predict(X_test)
print(f"决策树的准确率：{accuracy_score(y_test, y_pred):.3f}")


# 使用sklearn实现AdaBoost算法建模（基分类器是决策树）
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(base_estimator=base_model,
                            n_estimators=50,
                            learning_rate=0.5,
                            algorithm='SAMME.R',
                            random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"AdaBoost的准确率：{accuracy_score(y_test,y_pred):.3f}")

# 测试估计器个数的影响（n_estimators）
x = list(range(2, 102, 2))
y = []

for i in x:
    model = AdaBoostClassifier(base_estimator=base_model,
                               n_estimators=i,
                               learning_rate=0.5,
                               algorithm='SAMME.R',
                               random_state=1)

    model.fit(X_train, y_train)
    model_test_sc = accuracy_score(y_test, model.predict(X_test))
    y.append(model_test_sc)

plt.style.use('ggplot')
plt.title("Effect of n_estimators", pad=20)
plt.xlabel("Number of base estimators")
plt.ylabel("Test accuracy of AdaBoost")
plt.plot(x, y)
plt.show()


# 测试学习率的影响（learning_rate）







