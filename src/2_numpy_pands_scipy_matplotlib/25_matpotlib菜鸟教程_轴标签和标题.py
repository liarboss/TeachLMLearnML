# -*- encoding: utf-8 -*-
"""
@File    : 25_matpotlib菜鸟教程_轴标签和标题.py
@Time    : 2022/4/18 0018 9:56
@Author  : L
@Software: PyCharm
"""

# 我们可以使用 xlabel() 和 ylabel() 方法来设置 x 轴和 y 轴的标签。
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
plt.plot(x, y)
plt.xlabel("x - label")
plt.ylabel("y - label")

plt.show()

# 标题
# 我们可以使用 title() 方法来设置标题。
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
plt.plot(x, y)

plt.title("RUNOOB TEST TITLE")
plt.xlabel("x - label")
plt.ylabel("y - label")

plt.show()


