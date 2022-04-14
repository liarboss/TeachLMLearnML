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
x = [2, 3, 4]
y = [4, 6, 8]

plt.plot(x, y)
plt.show()





