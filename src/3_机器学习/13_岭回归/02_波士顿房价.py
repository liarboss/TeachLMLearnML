# -*- encoding: utf-8 -*-
"""
@File    : 02_波士顿房价.py
@Time    : 2022/5/5 0005 15:57
@Author  : L
@Software: PyCharm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import fetch_california_housing as fch, load_boston

# 1.加载数据集
boston_data = load_boston()
x = pd.DataFrame(boston_data.data)
y = boston_data.target
x.columns = boston_data.feature_names

# 2.划分训练集、测试集
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=420)
for i in [xtrain,xtest]:
    i.index = range(i.shape[0])

# 3.训练岭回归模型
reg = Ridge(alpha=1).fit(xtrain,ytrain)
# 3.1 R2指数
r2_score = reg.score(xtest,ytest) # 0.7504796922136654

# 4.探索交叉验证下不同alpha，岭回归与线性回归的结果变化
alpha_range = np.arange(1,1001,100)
ridge,lr = [],[]
for alpha in alpha_range:
    reg = Ridge(alpha=alpha)
    linear = LinearRegression()
    regs = cross_val_score(reg,x,y,cv=5,scoring='r2').mean()
    linears = cross_val_score(linear,x,y,cv=5,scoring='r2').mean()
    ridge.append(regs)
    lr.append(linears)
plt.plot(alpha_range,ridge,c='red',label='Ridge')
plt.plot(alpha_range,lr,c='orange',label='LR')
plt.title('Mean')
plt.legend()
plt.show()


alpha_range = np.arange(100,301,10)
ridge,lr = [],[]
for alpha in alpha_range:
    reg = Ridge(alpha=alpha)
    #linear = LinearRegression()
    regs = cross_val_score(reg,x,y,cv=5,scoring='r2').mean()
    #linears = cross_val_score(linear,x,y,cv=5,scoring='r2').mean()
    ridge.append(regs)
    lr.append(linears)
plt.plot(alpha_range,ridge,c='red',label='Ridge')
#plt.plot(alpha_range,lr,c='orange',label='LR')
plt.title('Mean')
plt.legend()
plt.show()



m = []
for i in range(100,300,10):
    # 3.训练岭回归模型
    reg = Ridge(alpha=i).fit(xtrain,ytrain)
    # 3.1 R2指数
    r2_score = reg.score(xtest,ytest) # 0.7504796922136654
    m.append(r2_score)
[0.7323071414013687, 0.7308298647680634, 0.7294076830146169, 0.7280399168133551, 0.7267248945259315, 0.7254603886919283,
 0.7242438916975088, 0.7230727895192606, 0.7219444701456151, 0.7208563898020044, 0.7198061117958099, 0.7187913275830451,
 0.7178098663372833, 0.716859697157614, 0.7159389266530864, 0.7150457937188373, 0.714178662706431, 0.7133360157813282,
 0.7125164449852812, 0.7117186443361565]


alpha_range = np.arange(1,1001,100)
ridge,lr = [],[]
for alpha in alpha_range:
    reg = Ridge(alpha=alpha)
    linear = LinearRegression()
    regs = cross_val_score(reg,x,y,cv=5,scoring='r2').var()
    linears = cross_val_score(linear,x,y,cv=5,scoring='r2').var()
    ridge.append(regs)
    lr.append(linears)
plt.plot(alpha_range,ridge,c='red',label='Ridge')
plt.plot(alpha_range,lr,c='orange',label='LR')
plt.title('Var')
plt.legend()
plt.show()
