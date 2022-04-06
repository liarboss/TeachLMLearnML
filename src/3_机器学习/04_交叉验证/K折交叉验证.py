#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
# @Time : 2022/4/6 17:38 
# @Author : cl
# @File : K折交叉验证.py 
# @Software: PyCharm
"""

"""
K折交叉验证： sklearn.model_selection.KFold(n_splits=3,shuffle=False,random_state=None)
思路：将训练/测试数据划分n_splits个互斥子集，每次用其中一个子集当作验证集，剩下的n_splits-1个作为训练集，进行n_splits次训练和测试，得到n_splits个结果
注意：对于不能均等分的数据集，前n_samples%n_splits子集拥有n_samples//n_splits+1个样本，其余子集只有n_samples//n_splits个样本

参数：
n_splits:表示划分几等份
shuffle:在每次划分时，是否进行洗牌
1)若为False，其效果等同于random_state等于整数，每次划分的结果相同
2）若为True时，每次划分的结果都不一样，表示经过洗牌，随机取样的
 random_state：随机种子数

属性：
①get_n_splits(X=None, y=None, groups=None)：获取参数n_splits的值
②split(X, y=None, groups=None)：将数据集划分成训练集和测试集，返回索引生成器
"""

import numpy as np
from sklearn.model_selection import KFold

# 设置shuffle=False，运行两次，发现两次结果相同
X = np.arange(24).reshape(12, 2)
y = np.random.choice([1, 2], 12, p=[0.4, 0.6])  # 1,2 总共出现12次，其中1出现的概率为0.4,2出现的概率为0.6
print(X)
print(y)
kf = KFold(n_splits=5, shuffle=False)
for train_index, test_index in kf.split(X):
    print('train_index %s, test_index %s' % (train_index, test_index))

X = np.arange(24).reshape(12, 2)
y = np.random.choice([1, 2], 12, p=[0.4, 0.6])  # 1,2 总共出现12次，其中1出现的概率为0.4,2出现的概率为0.6
kf = KFold(n_splits=5, shuffle=False)
for train_index, test_index in kf.split(X):
    print('train_index %s, test_index %s' % (train_index, test_index))

# B. 设置shuffle=True时，运行两次，发现两次运行的结果不同
from sklearn.model_selection import KFold
import numpy as np

print('#########################################################################')

# 设置shuffle=False，运行两次，发现两次结果相同
X = np.arange(24).reshape(12, 2)
y = np.random.choice([1, 2], 12, p=[0.4, 0.6])  # 1,2 总共出现12次，其中1出现的概率为0.4,2出现的概率为0.6
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    print('train_index %s, test_index %s' % (train_index, test_index))
print('-----------------------------------------------------------')
X = np.arange(24).reshape(12, 2)
y = np.random.choice([1, 2], 12, p=[0.4, 0.6])  # 1,2 总共出现12次，其中1出现的概率为0.4,2出现的概率为0.6
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    print('train_index %s, test_index %s' % (train_index, test_index))

# C: 设置shuffle=True和random_state=整数，发现每次运行的结果都相同
from sklearn.model_selection import KFold
import numpy as np
print('#########################################################################')


# 设置shuffle=False，运行两次，发现两次结果相同
X = np.arange(24).reshape(12, 2)
y = np.random.choice([1, 2], 12, p=[0.4, 0.6])  # 1,2 总共出现12次，其中1出现的概率为0.4,2出现的概率为0.6
kf = KFold(n_splits=5, shuffle=True, random_state=10)
for train_index, test_index in kf.split(X):
    print('train_index %s, test_index %s' % (train_index, test_index))
print('-----------------------------------------------------------')
X = np.arange(24).reshape(12, 2)
y = np.random.choice([1, 2], 12, p=[0.4, 0.6])  # 1,2 总共出现12次，其中1出现的概率为0.4,2出现的概率为0.6
kf = KFold(n_splits=5, shuffle=True, random_state=10)
for train_index, test_index in kf.split(X):
    print('train_index %s, test_index %s' % (train_index, test_index))
