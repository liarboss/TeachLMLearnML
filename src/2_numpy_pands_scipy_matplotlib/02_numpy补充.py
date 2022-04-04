#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
# @Time : 2022/4/4 11:44 
# @Author : cl
# @File : 02_numpy补充.py 
# @Software: PyCharm
"""

import numpy as np

"""
np.mat  矩阵
"""

m = np.mat([1, 2, 3])
print(m)
print(m[0, 1])  # 第一行，第2个数据

# print(m[0][1])  # 注意不能像数组那样取值了

# 将Python的列表转换成NumPy的矩阵

list = [1, 2, 3]
print(np.mat(list))

# Numpy dnarray转换成Numpy矩阵
n = np.array([1, 2, 3])
print(np.mat(n))

m = np.mat([[2, 5, 1], [4, 6, 2]])  # 创建2行3列矩阵
m.sort()
print(m)  # 对每一行进行排序

print(m.shape)  # 获得矩阵的行列数
m.shape[0]  # 获得矩阵的行数
m.shape[1]  # 获得矩阵的列数

m[1, :]  # 取得第一行的所有元素
m[1, 0:1]  # 第一行第0个元素，注意左闭右开
m[1, 0:3]
m[1, 0:2]

"""
矩阵乘法
"""
a = np.mat([[1, 2, 3], [2, 3, 4]])
b = np.mat([[1, 2], [3, 4], [5, 6]])
print(a)
print(b)
print(a * b)  # 方法一
print(np.matmul(a, b))  # 方法二
print(np.dot(a, b))  # 方法三

"""
multiply方法
"""

a = np.mat([[1, 2], [3, 4]])
b = np.mat([[2, 2], [3, 3]])
np.multiply(a, b)


"""
矩阵转置
"""

print(a)
print(a.T)
print(np.transpose(a))

## 矩阵求逆
print(a)
print(a.I)

print(a*a.I)