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

"""
NumPy 广播(Broadcast)
广播(Broadcast)是 numpy 对不同形状(shape)的数组进行数值计算的方式， 对数组的算术运算通常在相应的元素上进行。
如果两个数组 a 和 b 形状相同，即满足 a.shape == b.shape，那么 a*b 的结果就是 a 与 b 数组对应位相乘。这要求维数相同，且各维度的长度相同。
"""

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
c = a * b
print(c)

## 当运算中的 2 个数组的形状不同时，numpy 将自动触发广播机制
a = np.array([[0, 0, 0],
              [10, 10, 10],
              [20, 20, 20],
              [30, 30, 30]])
b = np.array([1, 2, 3])
print(a + b)

## 4x3 的二维数组与长为 3 的一维数组相加，等效于把数组 b 在二维上重复 4 次再运算：
a = np.array([[0, 0, 0],
              [10, 10, 10],
              [20, 20, 20],
              [30, 30, 30]])
b = np.array([1, 2, 3])
bb = np.tile(b, (4, 1))  # 重复 b 的各个维度
print(a + bb)

"""
广播的规则:

让所有输入数组都向其中形状最长的数组看齐，形状中不足的部分都通过在前面加 1 补齐。
输出数组的形状是输入数组形状的各个维度上的最大值。
如果输入数组的某个维度和输出数组的对应维度的长度相同或者其长度为 1 时，这个数组能够用来计算，否则出错。
当输入数组的某个维度的长度为 1 时，沿着此维度运算时都用此维度上的第一组值。
简单理解：对两个数组，分别比较他们的每一个维度（若其中一个数组没有当前维度则忽略），满足：

数组拥有相同形状。
当前维度的值相等。
当前维度的值有一个是 1。
"""

"""
NumPy 迭代数组
NumPy 迭代器对象 numpy.nditer 提供了一种灵活访问一个或者多个数组元素的方式。

迭代器最基本的任务的可以完成对数组元素的访问。
"""

a = np.arange(6).reshape(2, 3)
print('原始数组是：')
print(a)
print('\n')
print('迭代输出元素：')
for x in np.nditer(a):
    print(x, end=", ")
print('\n')

"""
以上实例不是使用标准 C 或者 Fortran 顺序，选择的顺序是和数组内存布局一致的，这样做是为了提升访问的效率，默认是行序优先（row-major order，或者说是 C-order）。
这反映了默认情况下只需访问每个元素，而无需考虑其特定顺序。我们可以通过迭代上述数组的转置来看到这一点，并与以 C 顺序访问数组转置的 copy 方式做对比，如下实例：
"""

## 按照内存结构迭代输出内容
# 从上述例子可以看出，a 和 a.T 的遍历顺序是一样的，也就是他们在内存中的存储顺序也是一样的，
# 但是 a.T.copy(order = 'C') 的遍历结果是不同的，那是因为它和前两种的存储方式是不一样的，默认是按行访问。
a = np.arange(6).reshape(2, 3)
for x in np.nditer(a.T):
    print(x, end=", ")
print('\n')

for x in np.nditer(a.T.copy(order='C')):
    print(x, end=", ")
print('\n')

"""
控制遍历顺序
for x in np.nditer(a, order='F'):Fortran order，即是列序优先；
for x in np.nditer(a.T, order='C'):C order，即是行序优先；
"""

a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print('原始数组是：')
print(a)
print('\n')
print('原始数组的转置是：')
b = a.T
print(b)
print('\n')
print('以 C 风格顺序排序：')
c = b.copy(order='C')
print(c)
for x in np.nditer(c):
    print(x, end=", ")
print('\n')
print('以 F 风格顺序排序：')
c = b.copy(order='F')
print(c)
for x in np.nditer(c):
    print(x, end=", ")

# 可以通过显式设置，来强制 nditer 对象使用某种顺序：
a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print('原始数组是：')
print(a)
print('\n')
print('以 C 风格顺序排序：')
for x in np.nditer(a, order='C'):
    print(x, end=", ")
print('\n')
print('以 F 风格顺序排序：')
for x in np.nditer(a, order='F'):
    print(x, end=", ")

"""
修改数组中元素的值
nditer 对象有另一个可选参数 op_flags。 默认情况下，nditer 将视待迭代遍历的数组为只读对象（read-only），
为了在遍历数组的同时，实现对数组元素值得修改，必须指定 read-write 或者 write-only 的模式。
"""
a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print('原始数组是：')
print(a)
print('\n')
for x in np.nditer(a, op_flags=['readwrite']):
    x[...] = 2 * x
print('修改后的数组是：')
print(a)

"""
使用外部循环
nditer 类的构造器拥有 flags 参数，它可以接受下列值：

参数	描述
c_index	可以跟踪 C 顺序的索引
f_index	可以跟踪 Fortran 顺序的索引
multi_index	每次迭代可以跟踪一种索引类型
external_loop	给出的值是具有多个值的一维数组，而不是零维数组
"""

a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print('原始数组是：')
print(a)
print('\n')
print('修改后的数组是：')
for x in np.nditer(a, flags=['external_loop'], order='F'):
    print(x, end=", ")

"""
广播迭代
如果两个数组是可广播的，nditer 组合对象能够同时迭代它们。 假设数组 a 的维度为 3X4，数组 b 的维度为 1X4 ，则使用以下迭代器（数组 b 被广播到 a 的大小）。
"""

a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print('第一个数组为：')
print(a)
print('\n')
print('第二个数组为：')
b = np.array([1, 2, 3, 4], dtype=int)
print(b)
print('\n')
print('修改后的数组为：')
for x, y in np.nditer([a, b]):
    print("%d:%d" % (x, y), end=", ")

"""
Numpy 数组操作
Numpy 中包含了一些函数用于处理数组，大概可分为以下几类：
"""

"""
numpy.reshape
numpy.reshape 函数可以在不改变数据的条件下修改形状，格式如下：
numpy.reshape(arr, newshape, order='C')
"""
a = np.arange(8)
print('原始数组：')
print(a)
print('\n')

b = a.reshape(4, 2)
print('修改后的数组：')
print(b)

## numpy.ndarray.flat  numpy.ndarray.flat 是一个数组元素迭代器，实例如下:
a = np.arange(9).reshape(3, 3)
print('原始数组：')
for row in a:
    print(row)

# 对数组中每个元素都进行处理，可以使用flat属性，该属性是一个数组元素迭代器：
print('迭代后的数组：')
for element in a.flat:
    print(element)

"""
numpy.ndarray.flatten
numpy.ndarray.flatten 返回一份数组拷贝，对拷贝所做的修改不会影响原始数组，格式如下：
ndarray.flatten(order='C')
"""
a = np.arange(8).reshape(2, 4)

print('原数组：')
print(a)
print('\n')
# 默认按行

print('展开的数组：')
print(a.flatten())
print('\n')

print('以 F 风格顺序展开的数组：')
print(a.flatten(order='F'))

"""
numpy.ravel
numpy.ravel() 展平的数组元素，顺序通常是"C风格"，返回的是数组视图（view，有点类似 C/C++引用reference的意味），修改会影响原始数组。
该函数接收两个参数：
numpy.ravel(a, order='C')
参数说明：
order：'C' -- 按行，'F' -- 按列，'A' -- 原顺序，'K' -- 元素在内存中的出现顺序。
"""
a = np.arange(8).reshape(2, 4)

print('原数组：')
print(a)
print('\n')

print('调用 ravel 函数之后：')
print(a.ravel())
print('\n')

print('以 F 风格顺序调用 ravel 函数之后：')
print(a.ravel(order='F'))

"""
numpy.transpose
numpy.transpose 函数用于对换数组的维度，格式如下：
numpy.transpose(arr, axes)
"""
a = np.arange(12).reshape(3, 4)

print('原数组：')
print(a)
print('\n')

print('对换数组：')
print(np.transpose(a))

## numpy.ndarray.T 类似 numpy.transpose：

a = np.arange(12).reshape(3, 4)

print('原数组：')
print(a)
print('\n')

print('转置数组：')
print(a.T)

"""
numpy.rollaxis
numpy.rollaxis 函数向后滚动特定的轴到一个特定位置，格式如下：
numpy.rollaxis(arr, axis, start)
挺抽象的
"""
# 创建了三维的 ndarray
a = np.arange(8).reshape(2, 2, 2)

print('原数组：')
print(a)
print('获取数组中一个值：')
print(np.where(a == 6))
print(a[1, 1, 0])  # 为 6
print('\n')

# 将轴 2 滚动到轴 0（宽度到深度）

print('调用 rollaxis 函数：')
b = np.rollaxis(a, 2, 0)
print(b)
# 查看元素 a[1,1,0]，即 6 的坐标，变成 [0, 1, 1]
# 最后一个 0 移动到最前面
print(np.where(b == 6))
print('\n')

# 将轴 2 滚动到轴 1：（宽度到高度）

print('调用 rollaxis 函数：')
c = np.rollaxis(a, 2, 1)
print(c)
# 查看元素 a[1,1,0]，即 6 的坐标，变成 [1, 0, 1]
# 最后的 0 和 它前面的 1 对换位置
print(np.where(c == 6))
print('\n')

"""
numpy.swapaxes
numpy.swapaxes 函数用于交换数组的两个轴，格式如下：
numpy.swapaxes(arr, axis1, axis2)
"""

# 创建了三维的 ndarray
a = np.arange(8).reshape(2, 2, 2)

print('原数组：')
print(a)
print('\n')
# 现在交换轴 0（深度方向）到轴 2（宽度方向）

print('调用 swapaxes 函数后的数组：')
print(np.swapaxes(a, 2, 0))

"""
NumPy 位运算
NumPy "bitwise_" 开头的函数是位运算函数。
NumPy 位运算包括以下几个函数：

函数	描述
bitwise_and	对数组元素执行位与操作
bitwise_or	对数组元素执行位或操作
invert	按位取反
left_shift	向左移动二进制表示的位
right_shift	向右移动二进制表示的位
"""

## bitwise_and() 函数对数组中整数的二进制形式执行位与运算。
print('13 和 17 的二进制形式：')
a, b = 13, 17
print(bin(a), bin(b))
print('\n')

print('13 和 17 的位与：')
print(np.bitwise_and(13, 17))

## bitwise_or()函数对数组中整数的二进制形式执行位或运算。

a, b = 13, 17
print('13 和 17 的二进制形式：')
print(bin(a), bin(b))

print('13 和 17 的位或：')
print(np.bitwise_or(13, 17))

# invert() 函数对数组中整数进行位取反运算，即 0 变成 1，1 变成 0。
print('13 的位反转，其中 ndarray 的 dtype 是 uint8：')
print(np.invert(np.array([13], dtype=np.uint8)))
print('\n')
# 比较 13 和 242 的二进制表示，我们发现了位的反转

print('13 的二进制表示：')
print(np.binary_repr(13, width=8))
print('\n')

print('242 的二进制表示：')
print(np.binary_repr(242, width=8))

"""
NumPy 统计函数
NumPy 提供了很多统计函数，用于从数组中查找最小元素，最大元素，百分位标准差和方差等。 函数说明如下：
"""

"""
numpy.amin() 和 numpy.amax()
numpy.amin() 用于计算数组中的元素沿指定轴的最小值。
numpy.amax() 用于计算数组中的元素沿指定轴的最大值。
"""
a = np.array([[3, 7, 5], [8, 4, 3], [2, 4, 9]])
print('我们的数组是：')
print(a)
print('\n')
print('调用 amin() 函数：')
print(np.amin(a, 1))
print('\n')
print('再次调用 amin() 函数：')
print(np.amin(a, 0))
print('\n')
print('调用 amax() 函数：')
print(np.amax(a))
print('\n')
print('再次调用 amax() 函数：')
print(np.amax(a, axis=0))

## numpy.ptp()函数计算数组中元素最大值与最小值的差（最大值 - 最小值）。

a = np.array([[3, 7, 5], [8, 4, 3], [2, 4, 9]])
print('我们的数组是：')
print(a)
print('\n')
print('调用 ptp() 函数：')
print(np.ptp(a))
print('\n')
print('沿轴 1 调用 ptp() 函数：')
print(np.ptp(a, axis=1))
print('\n')
print('沿轴 0 调用 ptp() 函数：')
print(np.ptp(a, axis=0))

## numpy.percentile()
## 百分位数是统计中使用的度量，表示小于这个值的观察值的百分比。 函数numpy.percentile()接受以下参数。


a = np.array([[10, 7, 4], [3, 2, 1]])
print('我们的数组是：')
print(a)

print('调用 percentile() 函数：')
# 50% 的分位数，就是 a 里排序之后的中位数
print(np.percentile(a, 50))

# axis 为 0，在纵列上求
print(np.percentile(a, 50, axis=0))

# axis 为 1，在横行上求
print(np.percentile(a, 50, axis=1))

# 保持维度不变
print(np.percentile(a, 50, axis=1, keepdims=True))

## numpy.median()
## numpy.median() 函数用于计算数组 a 中元素的中位数（中值）

a = np.array([[30, 65, 70], [80, 95, 10], [50, 90, 60]])
print('我们的数组是：')
print(a)
print('\n')
print('调用 median() 函数：')
print(np.median(a))
print('\n')
print('沿轴 0 调用 median() 函数：')
print(np.median(a, axis=0))
print('\n')
print('沿轴 1 调用 median() 函数：')
print(np.median(a, axis=1))

## numpy.mean()
## numpy.mean() 函数返回数组中元素的算术平均值。 如果提供了轴，则沿其计算。

a = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
print('我们的数组是：')
print(a)
print('\n')
print('调用 mean() 函数：')
print(np.mean(a))
print('\n')
print('沿轴 0 调用 mean() 函数：')
print(np.mean(a, axis=0))
print('\n')
print('沿轴 1 调用 mean() 函数：')
print(np.mean(a, axis=1))

## numpy.average()
## numpy.average() 函数根据在另一个数组中给出的各自的权重计算数组中元素的加权平均值。

a = np.array([1, 2, 3, 4])
print('我们的数组是：')
print(a)
print('\n')
print('调用 average() 函数：')
print(np.average(a))
print('\n')
# 不指定权重时相当于 mean 函数
wts = np.array([4, 3, 2, 1])
print('再次调用 average() 函数：')
print(np.average(a, weights=wts))
print('\n')
# 如果 returned 参数设为 true，则返回权重的和
print('权重的和：')
print(np.average([1, 2, 3, 4], weights=[4, 3, 2, 1], returned=True))

## 在多维数组中，可以指定用于计算的轴。
a = np.arange(6).reshape(3, 2)
print('我们的数组是：')
print(a)
print('\n')
print('修改后的数组：')
wt = np.array([3, 5])
print(np.average(a, axis=1, weights=wt))
print('\n')
print('修改后的数组：')
print(np.average(a, axis=1, weights=wt, returned=True))

## 标准差
## 标准差是一组数据平均值分散程度的一种度量。

print(np.std([1, 2, 3, 4]))

## 方差
## 统计中的方差（样本方差）是每个样本值与全体样本值的平均数之差的平方值的平均数，即 mean((x - x.mean())** 2)。
print(np.var([1, 2, 3, 4]))
