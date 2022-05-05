# -*- encoding: utf-8 -*-
"""
@File    : 01_加利福尼亚房价.py
@Time    : 2022/5/5 0005 15:49
@Author  : L
@Software: PyCharm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import fetch_california_housing as fch

# 1.加载数据集
house_value = fch()
x = pd.DataFrame(house_value.data)  # 提取数据集
y = house_value.target  # 提取标签
x.columns = ["住户收入中位数", "房屋使用年代中位数", "平均房间数目", "平均卧室数目", "街区人口", "平均入住率", "街区的纬度", "街区的经度"]

# 2.划分训练集、测试集
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=420)
# 2.1训练集恢复索引
for i in [xtrain, xtest]:
    i.index = range(i.shape[0])

# 3.采用岭回归训练模型 --- 定义正则项系数
reg = Ridge(alpha=1).fit(xtrain, ytrain)

# 3.1 R2指数
r2_score = reg.score(xtest, ytest)  # 0.6043610352312276

# 4.探索交叉验证下，岭回归与线性回归的结果变化
alpha_range = np.arange(1, 1001, 100)
ridge, lr = [], []
for alpha in alpha_range:
    reg = Ridge(alpha=alpha)
    linear = LinearRegression()
    regs = cross_val_score(reg, x, y, cv=5, scoring='r2').mean()
    linears = cross_val_score(linear, x, y, cv=5, scoring="r2").mean()
    ridge.append(regs)
    lr.append(linears)
plt.plot(alpha_range, ridge, c='red', label='Ridge')
plt.plot(alpha_range, lr, c='orange', label='LR')
plt.title('Mean')
plt.legend()
plt.show()

# 细化学习曲线
alpha_range = np.arange(1, 201, 10)
ridge, lr = [], []
for alpha in alpha_range:
    reg = Ridge(alpha=alpha)
    linear = LinearRegression()
    regs = cross_val_score(reg, x, y, cv=5, scoring='r2').mean()
    linears = cross_val_score(linear, x, y, cv=5, scoring="r2").mean()
    ridge.append(regs)
    lr.append(linears)
plt.plot(alpha_range, ridge, c='red', label='Ridge')
plt.plot(alpha_range, lr, c='orange', label='LR')
plt.title('Mean')
plt.legend()
plt.show()

# 方差
alpha_range = np.arange(1,1001,100)
ridge,lr = [],[]
for alpha in alpha_range:
    reg = Ridge(alpha=alpha)
    linear = LinearRegression()
    regs = cross_val_score(reg,x,y,cv=5,scoring='r2').var()
    linears = cross_val_score(linear,x,y,cv=5,scoring="r2").var()
    ridge.append(regs)
    lr.append(linears)
plt.plot(alpha_range,ridge,c='red',label='Ridge')
plt.plot(alpha_range,lr,c='orange',label='LR')
plt.title('Mean')
plt.legend()
plt.show()
