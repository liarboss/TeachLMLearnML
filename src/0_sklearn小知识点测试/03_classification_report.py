# -*- encoding: utf-8 -*-
"""
@File    : 03_classification_report.py
@Time    : 2022/4/8 0008 13:24
@Author  : L
@Software: PyCharm
"""

"""
classification_report简介
sklearn中的classification_report函数用于显示主要分类指标的文本报告．在报告中显示每个类的精确度，召回率，F1值等信息。
主要参数:
y_true：1维数组，或标签指示器数组/稀疏矩阵，目标值。
y_pred：1维数组，或标签指示器数组/稀疏矩阵，分类器返回的估计值。
labels：array，shape = [n_labels]，报表中包含的标签索引的可选列表。
target_names：字符串列表，与标签匹配的可选显示名称（相同顺序）。
sample_weight：类似于shape = [n_samples]的数组，可选项，样本权重。
digits：int，输出浮点值的位数．
"""



from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))
