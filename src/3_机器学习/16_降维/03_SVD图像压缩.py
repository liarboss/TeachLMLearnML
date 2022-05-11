# -*- encoding: utf-8 -*-
"""
@File    : 03_SVD图像压缩.py
@Time    : 2022/5/11 0011 10:34
@Author  : L
@Software: PyCharm
"""
import cv2
import numpy as np
# 调整该值重复运行代码保存图像
k = 10
img = cv2.imdecode(np.fromfile('./1.jpg', dtype=np.uint8), 0)
u, w, v = np.linalg.svd(img)
u = u[:, :k]
w = np.diag(w)
w = w[:k, :k]
v = v[:k, :]
img = u.dot(w).dot(v)
cv2.imencode('.jpg', img)[1].tofile('k={}.jpg'.format(k))
