# -*- encoding: utf-8 -*-
"""
@File    : 01_线性回归.py
@Time    : 2022/4/1 0001 15:16
@Author  : L
@Software: PyCharm
"""

"""
    线性回归：通过构建线性模型来进行预测的一种回归算法
"""
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

## 关闭科学计数法
np.set_printoptions(suppress=True)

## 数据集导入
boston = load_boston()
print(boston.DESCR)
# print(boston.target)
# print(boston.feature_names)
print(boston.items())
print(boston.data.shape)

## 数据分割
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=33, test_size=0.25)
print(x_train[1:5])

## 数据标准化
# print(type(y_test))

ss_x = StandardScaler()
ss_y = StandardScaler()

x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)

y_train = ss_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
y_test = ss_y.transform(y_test.reshape(-1, 1)).reshape(-1)

print(x_train[1:5])

## 模型训练
# 线性回归模型:解析解求解
lr = LinearRegression()
lr.fit(x_train, y_train)

# SGD回归模型:梯度下降法
sgd = SGDRegressor()
sgd.fit(x_train, y_train)

# 预测结果
y_predict = lr.predict(x_test)  # 这个结果是标准化之后的结果，需要转换
y_predict_inverse = ss_y.inverse_transform(y_predict.reshape(1, -1))
print(y_predict_inverse)

y_predict_sgd = sgd.predict(x_test)
y_predict_sgd_inverse = ss_y.inverse_transform(y_predict_sgd.reshape(1, -1))  # 反归一化
print("sgd预测结果：", y_predict_sgd_inverse)

## 模型评估
# 自带评估器
print(lr.score(x_test, y_test))
print(sgd.score(x_test, y_test))


