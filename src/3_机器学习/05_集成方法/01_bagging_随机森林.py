# -*- encoding: utf-8 -*-
"""
@File    : 01_bagging_随机森林.py
@Time    : 2022/4/8 0008 11:08
@Author  : L
@Software: PyCharm
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 导入模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# 生成数据
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=5000, n_features=30,
                           n_informative=25, n_redundant=5,
                           n_classes=4, random_state=2021)
# summarize the dataset
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
print(X_train.shape, X_test.shape)

# BAgg
bag_clf = BaggingClassifier(DecisionTreeClassifier(),
                            n_estimators=500, bootstrap=True, n_jobs=-1, oob_score=True)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
target_names = ['class 0', 'class 1', 'class 2', "class 3"]
print(classification_report(y_test, y_pred, target_names=target_names))

# RF
rnd_clf = RandomForestClassifier(n_estimators=500,
                                 max_depth=20,
                                 n_jobs=-1, oob_score=True, random_state=10)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
print(classification_report(y_test, y_pred_rf, target_names=target_names))
