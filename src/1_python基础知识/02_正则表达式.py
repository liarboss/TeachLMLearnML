# -*- encoding: utf-8 -*-
"""
@File    : 02_正则表达式.py
@Time    : 2022/3/30 0030 9:41
@Author  : L
@Software: PyCharm
"""

"""
## re.match函数
## re.match 尝试从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match() 就返回 none。
"""
import re

print(re.match('www', 'www.runoob.com').span())  # 在起始位置匹配
print(re.match('com', 'www.runoob.com'))  # 不在起始位置匹配

line = "Cats are smarter than dogs"

matchObj = re.match(r'(.*) are (.*?) .*', line, re.M | re.I)

if matchObj:
    print("matchObj.group() : ", matchObj.group())
    print("matchObj.group(1) : ", matchObj.group(1))
    print("matchObj.group(2) : ", matchObj.group(2))
else:
    print("No match!!")

"""
## re.search方法
## re.search 扫描整个字符串并返回第一个成功的匹配。
"""

print(re.search('www', 'www.runoob.com').span())  # 在起始位置匹配
print(re.search('com', 'www.runoob.com').span())  # 不在起始位置匹配

line = "aaa Cats are smarter than dogs"

searchObj = re.search(r'(.*) are (.*?) .*', line, re.M | re.I)

if searchObj:
    print("searchObj.group() : ", searchObj.group())
    print("searchObj.group(1) : ", searchObj.group(1))
    print("searchObj.group(2) : ", searchObj.group(2))
else:
    print("Nothing found!!")

"""
re.match与re.search的区别
re.match只匹配字符串的开始，如果字符串开始不符合正则表达式，则匹配失败，函数返回None；而re.search匹配整个字符串，直到找到一个匹配。
"""
line = "Cats are smarter than dogs";

matchObj = re.match(r'dogs', line, re.M | re.I)
if matchObj:
    print("match --> matchObj.group() : ", matchObj.group())
else:
    print("No match!!")

matchObj = re.search(r'dogs', line, re.M | re.I)
if matchObj:
    print("search --> searchObj.group() : ", matchObj.group())
else:
    print("No match!!")

"""
检索和替换
Python 的 re 模块提供了re.sub用于替换字符串中的匹配项。
re.sub(pattern, repl, string, count=0, flags=0)

"""
phone = "2004-959-559 # 这是一个国外电话号码"

# 删除字符串中的 Python注释
num = re.sub(r'#.*$', "", phone)
print("电话号码是: ", num)

# 删除非数字(-)的字符串
num = re.sub(r'\D', "", phone)
print("电话号码是 : ", num)


# 将匹配的数字乘以 2
def double(matched):
    value = int(matched.group('value'))
    return str(value * 2)


s = 'A23G4HFD567'
print(re.sub('(?P<value>\d+)', double, s))

"""
re.compile 函数
compile 函数用于编译正则表达式，生成一个正则表达式（ Pattern ）对象，供 match() 和 search() 这两个函数使用。
语法格式为：
re.compile(pattern[, flags])
"""
pattern = re.compile(r'\d+')
m = pattern.match('one12twothree34four')
print(m)
m = pattern.match('one12twothree34four', 2, 10)
print(m)
m = pattern.match('one12twothree34four', 3, 10)
print(m)
print(m.group(0))
print(m.start(0))
print(m.end(0))
print(m.span(0))

pattern = re.compile(r'([a-z]+) ([a-z]+)', re.I)
m = pattern.match('Hello World Wide Web')
print(m)
print(m.group(0))
print(m.span(0))
print(m.group(1))
print(m.span(1))
print(m.group(2))
print(m.span(2))
print(m.groups())
# print(m.group(3))

"""
findall
在字符串中找到正则表达式所匹配的所有子串，并返回一个列表，如果有多个匹配模式，则返回元组列表，如果没有找到匹配的，则返回空列表。
注意： match 和 search 是匹配一次 findall 匹配所有。
语法格式为：
findall(string[, pos[, endpos]])
"""
pattern = re.compile(r'\d+')  # 查找数字
result1 = pattern.findall('runoob 123 google 456')
result2 = pattern.findall('run88oob123google456', 0, 10)

print(result1)
print(result2)

## 多个匹配模式，返回元组列表：
result = re.findall(r'(\w+)=(\d+)', 'set width=20 and height=10')
print(result)

"""
re.finditer
和 findall 类似，在字符串中找到正则表达式所匹配的所有子串，并把它们作为一个迭代器返回。
re.finditer(pattern, string, flags=0)
"""

it = re.finditer(r"\d+", "12a32bc43jf3")
for match in it:
    print(match.group())

"""
re.split
split 方法按照能够匹配的子串将字符串分割后返回列表，它的使用形式如下：
re.split(pattern, string[, maxsplit=0, flags=0])
"""

print(re.split('\W+', 'runoob, runoob, runoob.'))
print(re.split('(\W+)', ' runoob, runoob, runoob.'))
print(re.split('\W+', ' runoob, runoob, runoob.', 1))
print(re.split('a*', 'hello world'))   # 对于一个找不到匹配的字符串而言，split 不会对其作出分割
