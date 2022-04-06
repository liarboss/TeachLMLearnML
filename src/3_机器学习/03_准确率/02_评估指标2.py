#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
# @Time : 2022/4/5 17:46 
# @Author : cl
# @File : 02_评估指标2.py 
# @Software: PyCharm
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
## 1. 在digits数据集上训练模型
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

digits = datasets.load_digits()
X = digits.data
y = digits.target
y = [1 if label >= 5 else 0 for label in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 创建模型
svc_clf = SVC(probability=True)
lr_clf = LogisticRegression(solver='saga', max_iter=100)
dt_clf = DecisionTreeClassifier(min_samples_leaf=5, max_depth=8)
knn_clf = KNeighborsClassifier()

# 训练模型
svc_clf.fit(X_train, y_train)
lr_clf.fit(X_train, y_train)
dt_clf.fit(X_train, y_train)
knn_clf.fit(X_train, y_train)

## 2. 使用plot_roc_curve函数绘制ROC曲线
# 创建画布
fig, ax = plt.subplots()

# svc_roc = plot_roc_curve(svc_clf, X_test, y_test, ax=ax)
lr_clf_roc = plot_roc_curve(lr_clf, X_test, y_test, ax=ax)
dt_clf_roc = plot_roc_curve(dt_clf, X_test, y_test, ax=ax)
# knn_clf_roc = plot_roc_curve(knn_clf, X_test, y_test, ax=ax)

# 参照线
ax.plot([0, 1], [0, 1], linestyle='--', color='r')

## 3. 使用roc_curve函数绘制ROC曲线

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

# 模型预测
y_pred_svc = svc_clf.predict_proba(X_test)[:, 1]
y_pred_lr = lr_clf.predict_proba(X_test)[:, 1]
y_pred_dt = dt_clf.predict_proba(X_test)[:, 1]
y_pred_knn = knn_clf.predict_proba(X_test)[:, 1]
y_pred_rand = np.random.rand(len(X_test))  # 随机生成概率

fpr_svc, tpr_svc, thres_svc = roc_curve(y_test, y_pred_svc)
fpr_lr, tpr_lr, thres_lr = roc_curve(y_test, y_pred_lr)
fpr_dt, tpr_dt, thres_dt = roc_curve(y_test, y_pred_dt)
fpr_knn, tpr_knn, thres_knn = roc_curve(y_test, y_pred_knn)
fpr_rand, tpr_rand, thres_rand = roc_curve(y_test, y_pred_rand)

print("SVC的AUC为:", auc(fpr_svc, tpr_svc))
print("LogitReg的AUC为:", auc(fpr_lr, tpr_lr))
print("DecisionTree的AUC为:", auc(fpr_dt, tpr_dt))
print("kNN的AUC为:", auc(fpr_knn, tpr_knn))
print("随机的AUC为:", auc(fpr_rand, tpr_rand))

# 创建画布
fig, ax = plt.subplots()

# 自定义标签名称label=''
# ax.plot(fpr_svc,tpr_svc,linewidth=2,
#         label='Random (AUC={})'.format(str(round(auc(fpr_svc,tpr_svc),3))))
ax.plot(fpr_lr, tpr_lr, linewidth=2,
        label='Logistic Regression (AUC={})'.format(str(round(auc(fpr_lr, tpr_lr), 3))))
ax.plot(fpr_dt, tpr_dt, linewidth=2,
        label='Decision Tree (AUC={})'.format(str(round(auc(fpr_dt, tpr_dt), 3))))
# ax.plot(fpr_knn,tpr_knn,linewidth=2,
#         label='K Nearest Neibor (AUC={})'.format(str(round(auc(fpr_knn,tpr_knn),3))))
# ax.plot(fpr_rand,tpr_rand,linewidth=2,
#         label='Random (AUC={})'.format(str(round(auc(fpr_rand,tpr_rand),3))))
# 绘制对角线
ax.plot([0, 1], [0, 1], linestyle='--', color='grey')

# 调整字体大小
plt.legend(fontsize=12)

## 4. 使用precision_recall_curve函数绘制PR曲线

from sklearn.metrics import precision_recall_curve

precision, recall, threshold = precision_recall_curve(y_test, y_pred_lr, pos_label=1)
fig = plt.figure()
plt.plot(precision, recall, label='Logistic')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()