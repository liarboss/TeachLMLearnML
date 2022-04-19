# -*- encoding: utf-8 -*-
"""
@File    : 01_朴素贝叶斯.py
@Time    : 2022/4/19 0019 15:43
@Author  : L
@Software: PyCharm
	刮北风	闷热	多云	天气预报有雨	真下雨？
第一天	否	是	否	是	0
第二天	是	是	是	否	1
第三天	否	是	是	否	1
第四天	否	否	否	是	0
第五天	否	是	是	否	1
第六天	否	是	否	是	0
第七天	是	否	否	是	0
"""
import numpy as np

x = np.array([[0, 1, 0, 1],
              [1, 1, 1, 0],
              [0, 1, 1, 0],
              [0, 0, 0, 1],
              [0, 1, 1, 0],
              [0, 1, 0, 1],
              [1, 0, 0, 1]]
             )
y = np.array([0, 1, 1, 0, 1, 0, 0])
# 导入朴素贝叶斯
from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB()
clf.fit(x, y)
# 将下一天的情况输入模型
Next_Day = [[0, 0, 1, 0]]
pre = clf.predict(Next_Day)
pre2 = clf.predict_proba(Next_Day)
# 输出模型预测结果
print("预测结果为：", pre)
# 输出模型预测的分类概率
print("预测的概率为：", pre2)
