# -*- encoding: utf-8 -*-
"""
@File    : 21_matplotlib.py
@Time    : 2022/4/14 0014 13:28
@Author  : L
@Software: PyCharm
https://zhuanlan.zhihu.com/p/442483978
基本图形绘制：折线图、柱状图、直方图、双轴线图等
绘制小技巧：添加图例、标题、注释、颜色等
实战：股票趋势图和K线图制作
"""

"""
导入库
一般绘图的时候需要导入常见的库；在使用matplotlib绘制的时候还需要解决中文的问题
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# %matplotlib inline

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号“-”显示为
plt.rcParams['axes.unicode_minus'] = False

"""
折线图
"""
# x = [2, 3, 4]
# y = [4, 6, 8]
#
# plt.plot(x, y)
# plt.show()

"""
多折线图
"""
# # 默认参数
# x1 = np.array([1, 3, 5])
# y1 = x1 + 4
# plt.plot(x1, y1)
#
# # 第二条红线
# y2 = x1 * 2
# plt.plot(x1, y2, color="red", linewidth=3, linestyle="--")
# plt.show()

"""
柱状图
"""
# x = [2, 3, 4, 5, 6]
# y = [4, 6, 8, 10, 12]
#
# plt.bar(x, y)
# plt.show()

"""
散点图
"""
# x = np.random.rand(20)  # 0-1之间的20个随机数
# y = np.random.rand(20)
#
# plt.scatter(x, y)
# plt.show()

"""
直方图
"""
# # 随机生成1000个服从正态分布的数据，均值为0，标准差为1
# data = np.random.randn(1000)
#
# plt.hist(data, bins=40, edgecolor="black")
# plt.show()

"""
频率直方图
"""
# data = np.random.randn(1000)
#
# # 区别：加上参数density=1
# plt.hist(data, bins=40, density=1, edgecolor="black")
# plt.show()

"""
绘图技巧
技巧1：设置大小
"""
# plt.rcParams["figure.figsize"] = (8, 6)
#
# x = [2, 3, 4]
# y = [4, 6, 8]
#
# plt.plot(x, y)
# # 设置大小  8代表800像素
# plt.show()

"""
添加文字说明
"""
# x = [2, 3, 4]
# y = [4, 6, 8]
#
# plt.plot(x, y)
# # 添加标题和轴名称
# plt.title("Title")
# plt.xlabel("x axis")
# plt.ylabel("y axis")
#
# plt.show()

"""
修改线条样式
"""
# import numpy as np
# import matplotlib.pyplot as plt
#
# # %matplotlib inline
# x = np.arange(1, 8)
#
# plt.plot(x, marker='>')
# plt.plot(x + 4, marker='+')
# plt.plot(x * 2, marker='o')
# plt.show()

"""
添加注释
"""
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False
#
# x = [1, 2, 3, 4]
# y = [1, 4, 9, 16]
#
# plt.plot(x, y)
# plt.xlabel('x坐标轴')
# plt.ylabel('y坐标轴')
# plt.title('标题')
#
# # 添加注释
# plt.annotate('我是注释',
#              xy=(2, 5),
#              xytext=(2, 10),
#              arrowprops=dict(facecolor='black',
#                              shrink=0.01),
#              )
#
# plt.show()

"""
添加图例
"""
# 第一条
# x1 = np.array([1, 3, 5])
# y1 = x1 + 4
# plt.plot(x1, y1, label="y=x+4 ")
#
# # 第二条红线
# y2 = x1 * 2
# plt.plot(x1, y2,
#          color="red",
#          linewidth=3,
#          linestyle="--",
#          label="y=x*2")
#
# # 设定位置
# plt.legend(loc='upper left')
# plt.show()

"""
调整颜色
"""

# x = np.arange(1, 8)
#
# # 颜色的多种写法
# plt.plot(x, color='r')  # r表示red  g表示green  b表示blue
# plt.plot(x + 1, color='0.5')
# plt.plot(x + 2, color='#AF00FF')
# plt.plot(x + 3, color=(0.1, 0.2, 0.3))
# plt.show()

"""
设置双轴
"""
# 1
# x1 = np.array([1, 3, 5])
# y1 = 50 * x1 + 14
# plt.plot(x1, y1, label="y=50 * x + 4 ")
# plt.legend(loc='upper right')  # 图例位置
#
# # 重要代码：设置双轴
# plt.twinx()
#
# # 2
# y2 = -x1 * 20 + 3
# plt.plot(x1, y2, color="red",
#          linewidth=3,
#          linestyle="--",
#          label="y=-x * 20 + 3")
# plt.legend(loc='upper left')
#
# plt.show()

"""
旋转轴刻度
"""
# x = ["Monday", "Thursday", "Wednesday"]
# y = [4, 6, 8]
#
# plt.plot(x, y)
# plt.xticks(rotation=45)
#
# plt.show()

"""
绘制多图-方法1
"""
# import matplotlib.pyplot as plt
#
# # 绘制第1张子图：折线图
# ax1 = plt.subplot(221)
# plt.plot([1, 2, 3], [2, 4, 6])
#
# # 绘制第2张子图：柱形图
# ax2 = plt.subplot(222)
# plt.bar([1, 2, 3], [2, 4, 6])
#
# # 绘制第3张子图：散点图
# ax3 = plt.subplot(223)
# plt.scatter([1, 3, 5], [7, 9, 11])
#
# # 绘制第4张子图：直方图
# ax4 = plt.subplot(224)
# plt.hist([2, 5, 2, 8, 4])
#
# plt.show()

"""
绘制多图-方法2
subplots函数主要是两个参数：nrows表示行数，ncols表示列数；同时设置大小figsize。
函数返回的是画布fig和子图合集axes
"""
# fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10,6))
#
# # flatten表示将子图合集展开，得到每个子图
# ax1,ax2,ax3,ax4 = axes.flatten()
#
# ax1.plot([1, 2, 3], [2, 4, 6])
# ax2.bar([1, 2, 3], [2, 4, 6])
# ax3.scatter([1, 3, 5], [7, 9, 11])
# ax4.hist([2, 5, 2, 8, 4])
#
# plt.show()

"""
实战：绘制股票趋势图
"""
import tushare as ts

df = ts.get_k_data("000001", start="2020-05-08", end="2020-08-08")
print(df)
