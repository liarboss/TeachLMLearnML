# -*- encoding: utf-8 -*-
"""
@File    : 神经网络-sklearndoc.py
@Time    : 2022/4/13 0013 16:41
@Author  : L
@Software: PyCharm
"""

import warnings
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# 下载数据
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = X / 255.0

# 切分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.7)

mlp = MLPClassifier(
    hidden_layer_sizes=(40,),  # 元组形式,长度n_layers-2,默认(100,),第i元素表示第i个神经元的个数
    max_iter=8,  # 可选，默认200，最大迭代次数。
    alpha=1e-4,  # 可选的，默认0.0001,正则化项参数
    solver="sgd",  # {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
    verbose=10,
    random_state=1,
    learning_rate_init=0.2,  # 给定的恒定学习率
)

# 这个例子不收敛,因为资源使用限制
# 我们的持续集成基础设施,所以我们的警告
# 这里忽略它

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=0.5 * vmin, vmax=0.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()



