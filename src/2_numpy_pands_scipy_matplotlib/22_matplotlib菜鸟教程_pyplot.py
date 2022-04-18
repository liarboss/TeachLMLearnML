# -*- encoding: utf-8 -*-
"""
@File    : 22_matplotlib菜鸟教程_pyplot.py
@Time    : 2022/4/15 0015 16:30
@Author  : L
@Software: PyCharm
"""
"""
Matplotlib Pyplot
Pyplot 是 Matplotlib 的子库，提供了和 MATLAB 类似的绘图 API。
Pyplot 是常用的绘图模块，能很方便让用户绘制 2D 图表。
Pyplot 包含一系列绘图函数的相关函数，每个函数会对当前的图像进行一些修改，例如：给图像加上标记，生新的图像，在图像中产生新的绘图区域等等。
https://www.runoob.com/matplotlib/matplotlib-pyplot.html
"""

import matplotlib.pyplot as plt

# 以下实例，我们通过两个坐标 (0,0) 到 (6,100) 来绘制一条线:
import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([0, 6])
ypoints = np.array([0, 100])
plt.plot(xpoints, ypoints)
plt.show()

# 如果我们要绘制坐标 (1, 3) 到 (8, 10) 的线，我们就需要传递两个数组 [1, 8] 和 [3, 10] 给 plot 函数：
import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([1, 8])
ypoints = np.array([3, 10])
plt.plot(xpoints, ypoints)
plt.show()

# 如果我们只想绘制两个坐标点，而不是一条线，可以使用 o 参数，表示一个实心圈的标记
import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([1, 8])
ypoints = np.array([3, 10])
plt.plot(xpoints, ypoints, 'o')
plt.show()

# 绘制一条不规则线，坐标为 (1, 3) 、 (2, 8) 、(6, 1) 、(8, 10)，对应的两个数组为：[1, 2, 6, 8] 与 [3, 8, 1, 10]。

import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])
plt.plot(xpoints, ypoints)
plt.show()

# 如果我们不指定 x 轴上的点，则 x 会根据 y 的值来设置为 0, 1, 2, 3..N-1。
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 10])
plt.plot(ypoints)
plt.show()

# 以下实例我们绘制一个正弦和余弦图，在 plt.plot() 参数中包含两对 x,y 值，第一对是 x,y，这对应于正弦函数，第二对是 x,z，这对应于余弦函数。
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 4 * np.pi, 0.1)  # start,stop,step
y = np.sin(x)
z = np.cos(x)
plt.plot(x, y,'o', x, z,'o')
plt.show()
