#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
# @Time : 2022/4/4 11:42 
# @Author : cl
# @File : 11_pandas.py
# @Software: PyCharm
"""

"""
Pandas 数据结构 - Series
Pandas Series 类似表格中的一个列（column），类似于一维数组，可以保存任何数据类型。
pandas.Series( data, index, dtype, name, copy)
参数说明：
data：一组数据(ndarray 类型)。
index：数据索引标签，如果不指定，默认从 0 开始。
dtype：数据类型，默认会自己判断。
name：设置名称。
copy：拷贝数据，默认为 False。
"""
# 创建一个简单的 Series 实例：
import pandas as pd

a = [1, 2, 3]
myvar = pd.Series(a)
print(myvar)

# 如果没有指定索引，索引值就从 0 开始，我们可以根据索引值读取数据：
a = [1, 2, 3]
myvar = pd.Series(a)
print(myvar[1])

# 我们可以指定索引值，如下实例
a = ["Google", "Runoob", "Wiki"]
myvar = pd.Series(a, index=["x", "y", "z"])
print(myvar)

# 根据索引值读取数据:
a = ["Google", "Runoob", "Wiki"]
myvar = pd.Series(a, index=["x", "y", "z"])
print(myvar["y"])

# 我们也可以使用 key/value 对象，类似字典来创建 Series：
sites = {1: "Google", 2: "Runoob", 3: "Wiki"}
myvar = pd.Series(sites)
print(myvar)

# 如果我们只需要字典中的一部分数据，只需要指定需要数据的索引即可，如下实例：
sites = {1: "Google", 2: "Runoob", 3: "Wiki"}
myvar = pd.Series(sites, index=[1, 2])
print(myvar)

# 设置 Series 名称参数：
sites = {1: "Google", 2: "Runoob", 3: "Wiki"}
myvar = pd.Series(sites, index=[1, 2], name="RUNOOB-Series-TEST")
print(myvar)

"""
Pandas 数据结构 - DataFrame
DataFrame 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）。
pandas.DataFrame( data, index, columns, dtype, copy)
参数说明：
data：一组数据(ndarray、series, map, lists, dict 等类型)。
index：索引值，或者可以称为行标签。
columns：列标签，默认为 RangeIndex (0, 1, 2, …, n) 。
dtype：数据类型。
copy：拷贝数据，默认为 False。
"""

# 实例 - 使用列表创建
data = [['Google', 10], ['Runoob', 12], ['Wiki', 13]]
df = pd.DataFrame(data, columns=['Site', 'Age'], dtype=float)
print(df)

# 实例 - 使用 ndarrays 创建
data = {'Site': ['Google', 'Runoob', 'Wiki'], 'Age': [10, 12, 13]}
df = pd.DataFrame(data)
print(df)

# 实例 - 使用字典创建
data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
print(df)

# Pandas 可以使用 loc 属性返回指定行的数据，如果没有设置索引，第一行索引为 0，第二行索引为 1
data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45]
}
# 数据载入到 DataFrame 对象
df = pd.DataFrame(data)
print(df)
# 返回第一行
print(df.loc[0])
# 返回第二行
print(df.loc[1])

# 也可以返回多行数据，使用 [[ ... ]] 格式，... 为各行的索引，以逗号隔开
data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45]
}

# 数据载入到 DataFrame 对象
df = pd.DataFrame(data)

# 返回第一行和第二行
print(df.loc[[0, 1]])

## 我们可以指定索引值，如下实例
data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45]
}
df = pd.DataFrame(data, index=["day1", "day2", "day3"])
print(df)

## Pandas 可以使用 loc 属性返回指定索引对应到某一行：
data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45]
}

df = pd.DataFrame(data, index=["day1", "day2", "day3"])

# 指定索引
print(df.loc["day2"])

"""
Pandas CSV 文件
CSV（Comma-Separated Values，逗号分隔值，有时也称为字符分隔值，因为分隔字符也可以不是逗号），其文件以纯文本形式存储表格数据（数字和文本）。
"""
df = pd.read_csv('nba.csv')
print(df.to_string())

# to_string() 用于返回 DataFrame 类型的数据，如果不使用该函数，则输出结果为数据的前面 5 行和末尾 5 行
df = pd.read_csv('nba.csv')
print(df)

# 我们也可以使用 to_csv() 方法将 DataFrame 存储为 csv 文件：

# 三个字段 name, site, age
nme = ["Google", "Runoob", "Taobao", "Wiki"]
st = ["www.google.com", "www.runoob.com", "www.taobao.com", "www.wikipedia.org"]
ag = [90, 40, 80, 98]

# 字典
dict = {'name': nme, 'site': st, 'age': ag}

df = pd.DataFrame(dict)

# 保存 dataframe
df.to_csv('site.csv')

"""
数据处理
"""
#  读取前面 5 行
df = pd.read_csv('nba.csv')
print(df.head())

# 读取前面 10 行
df = pd.read_csv('nba.csv')
print(df.head(10))

# tail( n ) 方法用于读取尾部的 n 行，如果不填参数 n ，默认返回 5 行，空行各个字段的值返回 NaN。
df = pd.read_csv('nba.csv')
print(df.tail())

# 读取末尾 10 行
df = pd.read_csv('nba.csv')
print(df.tail(10))

"""
info()
info() 方法返回表格的一些基本信息：
"""
df = pd.read_csv('nba.csv')
print(df.info())

"""
Pandas JSON
JSON（JavaScript Object Notation，JavaScript 对象表示法），是存储和交换文本信息的语法，类似 XML。
"""
df = pd.read_json('sites.json')
print(df.to_string())

# to_string() 用于返回 DataFrame 类型的数据，我们也可以直接处理 JSON 字符串。

data = [
    {
        "id": "A001",
        "name": "菜鸟教程",
        "url": "www.runoob.com",
        "likes": 61
    },
    {
        "id": "A002",
        "name": "Google",
        "url": "www.google.com",
        "likes": 124
    },
    {
        "id": "A003",
        "name": "淘宝",
        "url": "www.taobao.com",
        "likes": 45
    }
]
df = pd.DataFrame(data)
print(df)

# JSON 对象与 Python 字典具有相同的格式，所以我们可以直接将 Python 字典转化为 DataFrame 数据：
# 字典格式的 JSON
s = {
    "col1": {"row1": 1, "row2": 2, "row3": 3},
    "col2": {"row1": "x", "row2": "y", "row3": "z"}
}

# 读取 JSON 转为 DataFrame
df = pd.DataFrame(s)
print(df)

# 从 URL 中读取 JSON 数据：
URL = 'https://static.runoob.com/download/sites.json'
df = pd.read_json(URL)
print(df)

# 假设有一组内嵌的 JSON 数据文件 nested_list.json ：
df = pd.read_json('nested_list.json')
print(df)

# 这时我们就需要使用到 json_normalize() 方法将内嵌的数据完整的解析出来：
# 使用 Python JSON 模块载入数据
import json

with open('nested_list.json', 'r') as f:
    data = json.loads(f.read())

# 展平数据
df_nested_list = pd.json_normalize(data, record_path=['students'])
print(df_nested_list)

# 显示结果还没有包含 school_name 和 class 元素，如果需要展示出来可以使用 meta 参数来显示这些元数据：
# 使用 Python JSON 模块载入数据
with open('nested_list.json', 'r') as f:
    data = json.loads(f.read())

# 展平数据
df_nested_list = pd.json_normalize(
    data,
    record_path=['students'],
    meta=['school_name', 'class']
)
print(df_nested_list)

# 更复杂的json nested_mix.json 文件转换为 DataFrame：
# 使用 Python JSON 模块载入数据
with open('nested_mix.json', 'r') as f:
    data = json.loads(f.read())

df = pd.json_normalize(
    data,
    record_path=['students'],
    meta=[
        'class',
        ['info', 'president'],
        ['info', 'contacts', 'tel']
    ]
)

print(df)

# 这里我们需要使用到 glom 模块来处理数据套嵌，glom 模块允许我们使用 . 来访问内嵌对象的属性。
from glom import glom

df = pd.read_json('nested_deep.json')

data = df['students'].apply(lambda row: glom(row, 'grade.math'))
print(data)

"""
Pandas 数据清洗
数据清洗是对一些没有用的数据进行处理的过程。
很多数据集存在数据缺失、数据格式错误、错误数据或重复数据的情况，如果要对使数据分析更加准确，就需要对这些没有用的数据进行处理。
在这个教程中，我们将利用 Pandas包来进行数据清洗。
"""

"""
Pandas 清洗空值
如果我们要删除包含空字段的行，可以使用 dropna() 方法，语法格式如下：

DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
参数说明：

axis：默认为 0，表示逢空值剔除整行，如果设置参数 axis＝1 表示逢空值去掉整列。
how：默认为 'any' 如果一行（或一列）里任何一个数据有出现 NA 就去掉整行，如果设置 how='all' 一行（或列）都是 NA 才去掉这整行。
thresh：设置需要多少非空值的数据才可以保留下来的。
subset：设置想要检查的列。如果是多个列，可以使用列名的 list 作为参数。
inplace：如果设置 True，将计算得到的值直接覆盖之前的值并返回 None，修改的是源数据。
"""

# 我们可以通过 isnull() 判断各个单元格是否为空。
df = pd.read_csv('property-data.csv')
print(df)
print(df['NUM_BEDROOMS'])
print(df['NUM_BEDROOMS'].isnull())

# 以上例子中我们看到 Pandas 把 n/a 和 NA 当作空数据，na 不是空数据，不符合我们要求，我们可以指定空数据类型：
missing_values = ["n/a", "na", "--"]
df = pd.read_csv('property-data.csv', na_values=missing_values)

print(df['NUM_BEDROOMS'])
print(df['NUM_BEDROOMS'].isnull())

# 接下来的实例演示了删除包含空数据的行。
df = pd.read_csv('property-data.csv')
new_df = df.dropna()
print(new_df.to_string())

"""
注意：默认情况下，dropna() 方法返回一个新的 DataFrame，不会修改源数据。
如果你要修改源数据 DataFrame, 可以使用 inplace = True 参数:
"""

df = pd.read_csv('property-data.csv')
df.dropna(inplace=True)
print(df.to_string())

# 我们也可以移除指定列有空值的行：
# 移除 ST_NUM 列中字段值为空的行：

df = pd.read_csv('property-data.csv')
df.dropna(subset=['ST_NUM'], inplace=True)
print(df.to_string())

# 我们也可以 fillna() 方法来替换一些空字段：
# 使用 12345 替换空字段：

df = pd.read_csv('property-data.csv')
df.fillna(12345, inplace=True)
print(df.to_string())

# Pandas使用 mean()、median() 和 mode() 方法计算列的均值（所有值加起来的平均值）、中位数值（排序后排在中间的数）和众数（出现频率最高的数）。
df = pd.read_csv('property-data.csv')
x = df["ST_NUM"].mean()
df["ST_NUM"].fillna(x, inplace = True)
print(df.to_string())

# 使用 median() 方法计算列的中位数并替换空单元格：
df = pd.read_csv('property-data.csv')
x = df["ST_NUM"].median()
df["ST_NUM"].fillna(x, inplace = True)
print(df.to_string())

# 使用 mode() 方法计算列的众数并替换空单元格：
df = pd.read_csv('property-data.csv')
x = df["ST_NUM"].mode()
df["ST_NUM"].fillna(x, inplace = True)
print(df.to_string())


"""
Pandas 清洗格式错误数据
数据格式错误的单元格会使数据分析变得困难，甚至不可能。
我们可以通过包含空单元格的行，或者将列中的所有单元格转换为相同格式的数据。
"""
# 第三个日期格式错误
data = {
  "Date": ['2020/12/01', '2020/12/02' , '20201226'],
  "duration": [50, 40, 45]
}

df = pd.DataFrame(data, index = ["day1", "day2", "day3"])
df['Date'] = pd.to_datetime(df['Date'])
print(df.to_string())

"""
Pandas 清洗错误数据
数据错误也是很常见的情况，我们可以对错误的数据进行替换或移除。
"""
person = {
  "name": ['Google', 'Runoob' , 'Taobao'],
  "age": [50, 40, 12345]    # 12345 年龄数据是错误的
}

df = pd.DataFrame(person)
df.loc[2, 'age'] = 30 # 修改数据
print(df.to_string())

# 将 age 大于 120 的设置为 120:
person = {
  "name": ['Google', 'Runoob' , 'Taobao'],
  "age": [50, 200, 12345]
}

df = pd.DataFrame(person)

for x in df.index:
  if df.loc[x, "age"] > 120:
    df.loc[x, "age"] = 120

print(df.to_string())

# 将 age 大于 120 的删除:
person = {
  "name": ['Google', 'Runoob' , 'Taobao'],
  "age": [50, 40, 12345]    # 12345 年龄数据是错误的
}

df = pd.DataFrame(person)

for x in df.index:
  if df.loc[x, "age"] > 120:
    df.drop(x, inplace = True)

print(df.to_string())

# Pandas 清洗重复数据
# 如果我们要清洗重复数据，可以使用 duplicated() 和 drop_duplicates() 方法。

person = {
  "name": ['Google', 'Runoob', 'Runoob', 'Taobao'],
  "age": [50, 40, 40, 23]
}
df = pd.DataFrame(person)

print(df.duplicated())


## 删除重复数据，可以直接使用drop_duplicates() 方法。
persons = {
  "name": ['Google', 'Runoob', 'Runoob', 'Taobao'],
  "age": [50, 40, 40, 23]
}

df = pd.DataFrame(persons)

df.drop_duplicates(inplace = True)
print(df)

