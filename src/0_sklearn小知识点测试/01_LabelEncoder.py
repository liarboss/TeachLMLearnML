#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
# @Time : 2022/4/4 19:36 
# @Author : cl
# @File : 01_LabelEncoder.py
# @Software: PyCharm
"""
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])

print(list(le.classes_))
print(le.transform(["tokyo", "tokyo", "paris"]))
print(list(le.inverse_transform([2, 2, 1])))