# -*- encoding: utf-8 -*-
"""
@File    : 03_回归评估指标.py
@Time    : 2022/4/8 0008 10:20
@Author  : L
@Software: PyCharm
"""

"""
解释方差：最大1，越小模型越差，越大越好
"""
# explained_variance_score
from sklearn.metrics import explained_variance_score

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(explained_variance_score(y_true, y_pred))
y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
print(explained_variance_score(y_true, y_pred, multioutput="raw_values"))
print(explained_variance_score(y_true, y_pred, multioutput=[0.3, 0.7]))

# 结果
# 0.957173447537
# [ 0.96774194  1.        ]
# 0.990322580645

"""
平均绝对值误差 MAE：越小越好
"""

# mean_absolute_error
from sklearn.metrics import mean_absolute_error

y_true = [3, 0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(mean_absolute_error(y_true, y_pred))

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
print(mean_absolute_error(y_true, y_pred))
print(mean_absolute_error(y_true, y_pred, multioutput="raw_values"))
print(mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7]))

# 结果
# 0.5
# 0.75
# [ 0.5  1. ]
# 0.85

"""
均方误差：MSE ：越小越好
"""
# mean_squared_error
from sklearn.metrics import mean_squared_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(mean_squared_error(y_true, y_pred))
y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
print(mean_squared_error(y_true, y_pred))

"""
中值绝对误差(Median absolute error)  :此种方法非常适应含有离群点的数据集
"""

# median_absolute_error
from sklearn.metrics import median_absolute_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(median_absolute_error(y_true, y_pred))

# 结果
# 0.5

"""
可决系数R2：0-1之间，越大越好
"""

# r2_score
from sklearn.metrics import r2_score

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(r2_score(y_true, y_pred))

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
print(r2_score(y_true, y_pred, multioutput="variance_weighted"))

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
print(r2_score(y_true, y_pred, multioutput="uniform_average"))
print(r2_score(y_true, y_pred, multioutput="raw_values"))
print(r2_score(y_true, y_pred, multioutput=[0.3, 0.7]))

# 结果
# 0.948608137045
# 0.938256658596
# 0.936800526662
# [ 0.96543779  0.90816327]
# 0.92534562212
