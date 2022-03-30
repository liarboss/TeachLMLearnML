# -*- encoding: utf-8 -*-
"""
@File    : 01_推导式.py
@Time    : 2022/3/30 0030 9:37
@Author  : L
@Software: PyCharm
"""

"""
Python 支持各种数据结构的推导式：

列表(list)推导式
字典(dict)推导式
集合(set)推导式
元组(tuple)推导式
"""

"""
列表推导式list
[表达式 for 变量 in 列表] 
[out_exp_res for out_exp in input_list]
或者 
[表达式 for 变量 in 列表 if 条件]
[out_exp_res for out_exp in input_list if condition]
"""

## 长度4变大写
names = ["zhangsan", "lisi", "wangwu", "zhaoliu"]
upper_names = [name.upper() for name in names if len(name) == 4]
print(upper_names)

## 计算 30 以内可以被 3 整除的整数：
multiples = [i for i in range(30) if i % 3 == 0]
print(multiples)

"""
字典推导式map
字典推导基本格式：
{ key_expr: value_expr for value in collection }
或
{ key_expr: value_expr for value in collection if condition }
"""
listdemo = ["zhangsan", "lisi", "wangwu", "zhaoliu"]
# 将列表中各字符串值为键，各字符串的长度为值，组成键值对
newdict = {key: len(key) for key in listdemo}
print(newdict)

"""
集合推导式set
集合推导式基本格式：
{ expression for item in Sequence }
或
{ expression for item in Sequence if conditional }
"""
## 平方数
setnew = {i ** 2 for i in (1, 2, 3)}
print(setnew)

## 不是abc的数
a = {x for x in 'abracadabra' if x not in 'abc'}
print(a)

"""
元组推导式tuple
元组推导式可以利用 range 区间、元组、列表、字典和集合等数据类型，快速生成一个满足指定需求的元组。
元组推导式基本格式：
(expression for item in Sequence )
或
(expression for item in Sequence if conditional )
"""
## 元祖推导式返回的是generator，需要用tuple转换
a = (x for x in range(1, 10))
print(a)
print(tuple(a))
