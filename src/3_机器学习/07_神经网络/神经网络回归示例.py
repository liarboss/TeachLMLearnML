# -*- encoding: utf-8 -*-
"""
@File    : 神经网络回归示例.py
@Time    : 2022/4/13 0013 17:09
@Author  : L
@Software: PyCharm
"""

# 1. 数据集
import warnings
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor


boston = load_boston()
data = boston['data']  # shape=(506, 13)
target = boston['target']  # shape=(506,)
# print(boston['DESCR'])


# 2. 划分数据集
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
# 训练集和测试集的shape：(404, 13)与 (102, 13)

# 对训练集进行标准化，同时将标准化的规则应用到验证集
from sklearn.preprocessing import StandardScaler

std = StandardScaler()
train_data_std = std.fit_transform(train_data)
test_data_std = std.transform(test_data)

# 3. 构建模型
regr = MLPRegressor(solver='adam', hidden_layer_sizes=(50, 50), activation='tanh', max_iter=5000)
regr.fit(train_data_std,train_target)

print(regr.score(train_data_std, train_target))
y_pred = regr.predict(test_data_std)
# 4. 评价回归模型
"""
关于回归模型的评价指标如下：

方法名称	最优值	sklearn函数
平均绝对误差	0.0	sklearn.metrics.mean_absolute_error
均方误差	0.0	sklearn.metrics.mean_squared_error
中值绝对误差	0.0	sklearn.metrics.median_absolute_error
可解释方差值	1.0	sklearn.metrics.explained_variance_score
R 2 R^2R 
2
 值	1.0	sklearn.metrics.r2_score
 
 
"""
error = mean_absolute_error(test_target, y_pred)
print(error)  # 2.3726167664920306


plt.figure()
plt.plot(range(len(y_pred)), y_pred, color='blue')
plt.plot(range(len(y_pred)), test_target, color='red')
plt.show()
