# -*- encoding: utf-8 -*-
"""
@File    : knn.py
@Time    : 2022/5/7 0007 13:37
@Author  : L
@Software: PyCharm
"""

from sklearn import datasets  # 导入数据模块
from sklearn.model_selection import train_test_split  # 导入切分训练集、测试集模块
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

print(iris_x)
print(iris_y)

x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.3)
print(y_train)
print(y_test)

# #输出
# [1 0 0 1 1 1 2 2 0 2 1 1 1 1 1 1 2 1 0 0 2 2 0 1 2 2 1 2 0 2 2 0 0 1 1 2 1
#  2 0 0 1 0 0 0 2 2 0 0 2 1 0 0 0 1 1 1 0 0 2 0 0 2 2 1 0 2 2 1 2 1 1 0 1 2
#  2 2 0 2 1 2 2 0 2 1 2 2 1 2 1 2 0 1 2 2 0 1 2 1 1 0 0 0 2 1 0]
# [0 2 0 0 0 0 1 0 1 0 1 2 2 0 1 2 1 2 1 0 1 2 2 0 0 0 2 1 1 1 0 0 2 2 0 2 0
#  1 1 2 0 1 1 1 2]


knn = KNeighborsClassifier()  # 实例化KNN模型
knn.fit(x_train, y_train)  # 放入训练数据进行训练
print(knn.predict(x_test))  # 打印预测内容
print(y_test)  # 实际标签
# 可见只有一个没有预测正确
