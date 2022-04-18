# -*- encoding: utf-8 -*-
"""
@File    : 23_matplotlib菜鸟教程_绘图标记.py
@Time    : 2022/4/18 0018 8:40
@Author  : L
@Software: PyCharm
https://www.runoob.com/matplotlib/matplotlib-marker.html
"""
"""
Matplotlib 绘图标记
绘图过程如果我们想要给坐标自定义一些不一样的标记，就可以使用 plot() 方法的 marker 参数来定义。
"""

import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([1, 3, 4, 5, 8, 9, 6, 1, 3, 4, 5, 2, 4])

plt.plot(ypoints, marker='o')
plt.show()

# 以下实例定义了 * 标记：
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([1, 3, 4, 5, 8, 9, 6, 1, 3, 4, 5, 2, 4])
plt.plot(ypoints, marker='*')
plt.show()

# 以下实例定义了下箭头：
import matplotlib.pyplot as plt
import matplotlib.markers

plt.plot([1, 2, 3], marker=matplotlib.markers.CARETDOWNBASE)
plt.show()

"""
fmt 参数
fmt 参数定义了基本格式，如标记、线条样式和颜色。
fmt = '[marker][line][color]'
例如 o:r，o 表示实心圆标记，: 表示虚线，r 表示颜色为红色。
"""
# 例如 o:r，o 表示实心圆标记，: 表示虚线，r 表示颜色为红色。
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([6, 2, 13, 10])
plt.plot(ypoints, 'o:r')
plt.show()

"""
标记大小与颜色
我们可以自定义标记的大小与颜色，使用的参数分别是：
markersize，简写为 ms：定义标记的大小。
markerfacecolor，简写为 mfc：定义标记内部的颜色。
markeredgecolor，简写为 mec：定义标记边框的颜色。
"""
# 设置标记大小：
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([6, 2, 13, 10])
plt.plot(ypoints, marker='o', ms=20)
plt.show()

# 设置标记外边框颜色：

import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([6, 2, 13, 10])
plt.plot(ypoints, marker='o', ms=20, mec='r')
plt.show()

# 设置标记内部颜色：

import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([6, 2, 13, 10])
plt.plot(ypoints, marker='o', ms=20, mfc='r')
plt.show()

# 自定义标记内部与边框的颜色：
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([6, 2, 13, 10])
plt.plot(ypoints, marker='o', ms=20, mec='#4CAF50', mfc='#4CAF50')
plt.show()
