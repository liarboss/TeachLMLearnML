#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
# @Time : 2022/3/31 10:03 
# @Author : cl
# @File : 01_numpy.py 
# @Software: PyCharm
"""

"""
Ndarray 对象: N 维数组对象 ndarray
numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)

"""
import numpy as np

# 创建一维数组
a = np.array([1, 2, 3])
print(a)

# 多于一个维度
a = np.array([[1, 2], [3, 4]])
print(a)

# 最小维度
a = np.array([1, 2, 3, 4, 5], ndmin=2)
print(a)

# dtype 参数 复数？？？
a = np.array([1, 2, 3], dtype=complex)
print(a)

###########################################################
"""
NumPy 数据类型
https://www.runoob.com/numpy/numpy-dtype.html
"""
# 使用标量类型
dt = np.dtype(np.int32)
print(dt)

# int8, int16, int32, int64 四种数据类型可以使用字符串 'i1', 'i2','i4','i8' 代替
dt = np.dtype('i4')
print(dt)

# 字节顺序标注
dt = np.dtype('<i4')
print(dt)

# 首先创建结构化数据类型
dt = np.dtype([('age', np.int8)])
print(dt)

# 将数据类型应用于 ndarray 对象
dt = np.dtype([('age', np.int8)])
a = np.array([(10,), (20,), (30,)], dtype=dt)
print(a)

##下面的示例定义一个结构化数据类型 student，包含字符串字段 name，整数字段 age，及浮点字段 marks，并将这个 dtype 应用到 ndarray 对象。

student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
print(student)

student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
a = np.array([('abc', 21, 50), ('xyz', 18, 75)], dtype=student)
print(a)

"""
NumPy 数组属性
https://www.runoob.com/numpy/numpy-array-attributes.html
"""

## ndarray.ndim 用于返回数组的维数
a = np.arange(24)
print(a.ndim)  # a 现只有一个维度
# 现在调整其大小
b = a.reshape(2, 4, 3)  # b 现在拥有三个维度
print(b.ndim)

## ndarray.shape 表示数组的维度
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)

a = np.array([[1, 2, 3], [4, 5, 6]])
a.shape = (3, 2)
print(a)

a = np.array([[1, 2, 3], [4, 5, 6]])
b = a.reshape(3, 2)
print(b)

## ndarray.itemsize 以字节的形式返回数组中每一个元素的大小。

# 数组的 dtype 为 int8（一个字节）
x = np.array([1, 2, 3, 4, 5], dtype=np.int8)
print(x.itemsize)

# 数组的 dtype 现在为 float64（八个字节）
y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
print(y.itemsize)

"""
ndarray.flags 返回 ndarray 对象的内存信息

属性	描述
C_CONTIGUOUS (C)	数据是在一个单一的C风格的连续段中
F_CONTIGUOUS (F)	数据是在一个单一的Fortran风格的连续段中
OWNDATA (O)	数组拥有它所使用的内存或从另一个对象中借用它
WRITEABLE (W)	数据区域可以被写入，将该值设置为 False，则数据为只读
ALIGNED (A)	数据和所有元素都适当地对齐到硬件上
UPDATEIFCOPY (U)	这个数组是其它数组的一个副本，当这个数组被释放时，原数组的内容将被更新

https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html
"""

x = np.array([1, 2, 3, 4, 5])
print(x.flags)

"""
NumPy 创建数组
ndarray 数组除了可以使用底层 ndarray 构造器来创建外，也可以通过以下几种方式来创建。
"""

# numpy.empty
# numpy.empty(shape, dtype = float, order = 'C')
# 数组元素为随机值，因为它们未初始化
v = np.empty([3, 2], dtype=int)
print(v)

# numpy.zeros
# numpy.zeros(shape, dtype = float, order = 'C')
# 默认为浮点数
x = np.zeros(5)
print(x)

# 设置类型为整数
y = np.zeros((5,), dtype=np.int)
print(y)

# 自定义类型
z = np.zeros((2, 2), dtype=[('x', 'i4'), ('y', 'i4')])
print(z)

# numpy.ones
# numpy.ones(shape, dtype = None, order = 'C')

# 默认为浮点数
x = np.ones(5)
print(x)

# 自定义类型
x = np.ones([2, 2], dtype=int)
print(x)

"""
NumPy 从已有的数组创建数组
numpy.asarray(a, dtype = None, order = None)
"""

## 将列表转换为 ndarray:
x = [1, 2, 3]
a = np.asarray(x)
print(a)

## 将元组转换为 ndarray:
x = (1, 2, 3)
a = np.asarray(x)
print(a)

## 将元组列表转换为 ndarray:

x = [(1, 2, 3), (4, 5)]
a = np.asarray(x)
print(a)

## 设置了 dtype 参数：

x = [1, 2, 3]
a = np.asarray(x, dtype=float)
print(a)

"""
numpy.frombuffer
numpy.frombuffer 用于实现动态数组。

numpy.frombuffer 接受 buffer 输入参数，以流的形式读入转化成 ndarray 对象。
numpy.frombuffer(buffer, dtype = float, count = -1, offset = 0)
"""
s = b'Hello World'
a = np.frombuffer(s, dtype='S1')
print(a)

"""
numpy.fromiter
numpy.fromiter 方法从可迭代对象中建立 ndarray 对象，返回一维数组。
numpy.fromiter(iterable, dtype, count=-1)
"""

# 使用 range 函数创建列表对象
list = range(5)
it = iter(list)

# 使用迭代器创建 ndarray
x = np.fromiter(it, dtype=float)
print(x)

"""
NumPy 从数值范围创建数组
numpy.arange(start, stop, step, dtype)

"""

x = np.arange(5)
print(x)

# 设置了 dtype , 设置返回类型位 float:
x = np.arange(5, dtype=float)
print(x)

# 设置了起始值、终止值及步长：
x = np.arange(10, 20, 2)
print(x)

""" 
***********
numpy.linspace
numpy.linspace 函数用于创建一个一维数组，数组是一个<等差数列>构成的，格式如下：

np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
"""
a = np.linspace(1, 10, 10)
print(a)

## 设置元素全部是1的等差数列：
a = np.linspace(1, 1, 10)
print(a)

## 将 endpoint 设为 false，不包含终止值：

a = np.linspace(10, 20, 5, endpoint=False)
print(a)

a = np.linspace(1, 10, 20, retstep=True)

print(a)
# 拓展例子
b = np.linspace(1, 10, 10).reshape([10, 1])
print(b)

"""
numpy.logspace
numpy.logspace 函数用于创建一个于等比数列。格式如下：

np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
"""
# 默认底数是 10
a = np.logspace(1.0, 2.0, num=10, base=2)
print(a)

"""
NumPy 切片和索引
ndarray对象的内容可以通过索引或切片来访问和修改，与 Python 中 list 的切片操作一样。

"""

a = np.arange(10)
s = slice(2, 7, 2)  # 从索引 2 开始到索引 7 停止，间隔为2
print(a[s])

a = np.arange(10)
b = a[2:7:2]  # 从索引 2 开始到索引 7 停止，间隔为 2
print(b)

## 冒号 : 的解释：如果只放置一个参数，如 [2]，将返回与该索引相对应的单个元素。如果为 [2:]
# ，表示从该索引开始以后的所有项都将被提取。如果使用了两个参数，如 [2:7]，那么则提取两个索引(不包括停止索引)之间的项。
a = np.arange(10)  # [0 1 2 3 4 5 6 7 8 9]
b = a[5]
print(b)

a = np.arange(10)
print(a[2:])

a = np.arange(10)  # [0 1 2 3 4 5 6 7 8 9]
print(a[2:5])

a = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
print(a)
# 从某个索引处开始切割
print('从数组索引 a[1:] 处开始切割')
print(a[1:])

## 切片还可以包括省略号 …，来使选择元组的长度与数组的维度相同。 如果在行位置使用省略号，它将返回包含行中元素的 ndarray。

a = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
print(a[:])
print(a[..., 1])  # 第2列元素
print(a[1, ...])  # 第2行元素
print(a[..., 1:])  # 第2列及剩下的所有元素

"""
NumPy 高级索引
"""
# 以下实例获取数组中(0,0)，(1,1)和(2,0)位置处的元素。
x = np.array([[1, 2], [3, 4], [5, 6]])
y = x[[0, 1, 2], [0, 1, 0]]
print(y)

## 以下实例获取了 4X3 数组中的四个角的元素。 行索引是 [0,0] 和 [3,3]，而列索引是 [0,2] 和 [0,2]。

x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
print('我们的数组是：')
print(x)
print('\n')
rows = np.array([[0, 0], [3, 3]])
cols = np.array([[0, 2], [0, 2]])
y = x[rows, cols]
# y = x[[0, 0, 3, 3], [0, 2, 0, 2]]
print('这个数组的四个角元素是：')
print(y)

## 可以借助切片 : 或 … 与索引数组组合
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = a[1:3, 1:3]
c = a[1:3, [1, 2]]
d = a[..., 1:]
print(b)
print(c)
print(d)

"""
布尔索引
我们可以通过一个布尔数组来索引目标数组。
布尔索引通过布尔运算（如：比较运算符）来获取符合指定条件的元素的数组。
"""

x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
print('我们的数组是：')
print(x)
print('\n')
# 现在我们会打印出大于 5 的元素
print('大于 5 的元素是：')
print(x[x > 5])

# 以下实例使用了 ~（取补运算符）来过滤 NaN。
a = np.array([np.nan, 1, 2, np.nan, 3, 4, 5])
print(a[~np.isnan(a)])

# 以下实例演示如何从数组中过滤掉非复数元素。

a = np.array([1, 2 + 6j, 5, 3.5 + 5j])
print(a[np.iscomplex(a)])

"""
花式索引
花式索引指的是利用整数数组进行索引。
花式索引根据索引数组的值作为目标数组的某个轴的下标来取值。对于使用一维整型数组作为索引，如果目标是一维数组，那么索引的结果就是对应下标的行，如果目标是二维数组，那么就是对应位置的元素。
花式索引跟切片不一样，它总是将数据复制到新数组中。
"""

x = np.arange(32).reshape((8, 4))
print(x)
print(x[[4, 2, 1, 7]])

## 传入倒序索引数组
x = np.arange(32).reshape((8, 4))
print(x[[-4, -2, -1, -7]])

## 传入多个索引数组（要使用np.ix_）
x = np.arange(32).reshape((8, 4))
print(x[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])])
