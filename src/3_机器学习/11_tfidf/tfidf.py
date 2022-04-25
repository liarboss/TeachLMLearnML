# -*- encoding: utf-8 -*-
"""
@File    : tfidf.py
@Time    : 2022/4/25 0025 10:15
@Author  : L
@Software: PyCharm
"""

# TfidfVectorizer=TfidfTransformer + CountVectorizer
# fit_transform方法将语料转化成TF-IDF权重矩阵，get_feature_names方法可得到词汇表。


from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.shape)

# 将权重矩阵转化成array：
print(X.toarray())

# 可以看到是4行9列，m行n列处值的含义是词汇表中第n个词在第m篇文档的TF-IDF值。提取单篇文档的关键词只需要将矩阵按行的值从大到小排序取前几个即可。
# 如果要提取所有文档的关键词，我们可以将矩阵按列求和，得到每个词汇综合TF-IDF值。

print(X.toarray().sum(axis=0))

# 转化成dataframe，再排序。
import pandas as pd

data = {'word': vectorizer.get_feature_names(),
        'tfidf': X.toarray().sum(axis=0).tolist()}
df = pd.DataFrame(data)
print(df.sort_values(by="tfidf", ascending=False))
