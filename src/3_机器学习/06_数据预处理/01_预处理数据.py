# -*- encoding: utf-8 -*-
"""
@File    : 01_预处理数据.py
@Time    : 2022/4/11 0011 13:36
@Author  : L
@Software: PyCharm
"""

"""
5.3.1 标准化，也称去均值和方差按比例缩放
数据集的 标准化 对scikit-learn中实现的大多数机器学习算法来说是 常见的要求 。如果个别特征或多或少看起来不是很像标准正态分布(具有零均值和单位方差)，
那么它们的表现力可能会较差。
在实际情况中,我们经常忽略特征的分布形状，直接经过去均值来对某个特征进行中心化，再通过除以非常量特征(non-constant features)的标准差进行缩放。
例如，在机器学习算法的目标函数(例如SVM的RBF内核或线性模型的l1和l2正则化)，许多学习算法中目标函数的基础都是假设所有的特征都是零均值并且具有同一阶数上的方差。
如果某个特征的方差比其他特征大几个数量级，那么它就会在学习算法中占据主导位置，导致学习器并不能像我们说期望的那样，从其他特征中学习。
"""

from sklearn import preprocessing
import numpy as np

X_train = np.array([[1., -1., 2.],
                    [2., 0., 0.],
                    [0., 1., -1.]])
X_scaled = preprocessing.scale(X_train)

print(X_scaled)
# array([[ 0.  ..., -1.22...,  1.33...],
#        [ 1.22...,  0.  ..., -0.26...],
#        [-1.22...,  1.22..., -1.06...]])

# 经过缩放后的数据具有零均值以及标准方差:


print(X_scaled.mean(axis=0))
# array([ 0.,  0.,  0.])

print(X_scaled.std(axis=0))
# array([ 1.,  1.,  1.])

"""
预处理 模块还提供了一个实用类 StandardScaler ，它实现了转化器的API来计算训练集上的平均值和标准偏差，以便以后能够在测试集上重新应用相同的变换。
因此，这个类适用于 sklearn.pipeline.Pipeline 的早期步骤
"""

scaler = preprocessing.StandardScaler().fit(X_train)
print(scaler)
# StandardScaler(copy=True, with_mean=True, with_std=True)

print(scaler.mean_)
# array([ 1. ...,  0. ...,  0.33...])

print(scaler.scale_)  ##标准差
# array([ 0.81...,  0.81...,  1.24...])

print(scaler.transform(X_train))
# array([[ 0.  ..., -1.22...,  1.33...],
#  [ 1.22...,  0.  ..., -0.26...],
#  [-1.22...,  1.22..., -1.06...]])

"""
缩放类对象可以在新的数据上实现和训练集相同缩放操作:
"""
X_test = [[-1., 1., 0.]]
scaler.transform(X_test)
# array([[-2.44...,  1.22..., -0.26...]])


"""
5.3.1.1 将特征缩放至特定范围内
一种标准化是将特征缩放到给定的最小值和最大值之间，通常在零和一之间，或者也可以将每个特征的最大绝对值转换至单位大小。
可以分别使用 MinMaxScaler 和 MaxAbsScaler 实现。
使用这种缩放的目的包括实现特征极小方差的鲁棒性以及在稀疏矩阵中保留零元素。
"""

# 以下是一个将简单的数据矩阵缩放到[0, 1]的示例:
X_train = np.array([[1., -1., 2.],
                    [2., 0., 0.],
                    [0., 1., -1.]])

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_train_minmax
# array([[ 0.5       ,  0.        ,  1.        ],
#  [ 1.        ,  0.5       ,  0.33333333],
#  [ 0.        ,  1.        ,  0.        ]])

# 同样的转换实例可以被用与在训练过程中不可见的测试数据:实现和训练数据一致的缩放和移位操作
X_test = np.array([[-3., -1., 4.]])
X_test_minmax = min_max_scaler.transform(X_test)
X_test_minmax
# array([[-1.5       ,  0.        ,  1.66666667]])

# 可以检查缩放器（scaler）属性，来观察在训练集中学习到的转换操作的基本性质
print(min_max_scaler.scale_)
# array([ 0.5       ,  0.5       ,  0.33...])

print(min_max_scaler.min_)
# array([ 0.        ,  0.5       ,  0.33...])


"""
类 MaxAbsScaler 的工作原理非常相似，但是它只通过除以每个特征的最大值将训练数据特征缩放至 [-1, 1] 范围内，这就意味着，训练数据应该是已经零中心化或者是稀疏数据。 
示例::用先前示例的数据实现最大绝对值缩放操作。
"""

X_train = np.array([[1., -1., 2.],
                    [2., 0., 0.],
                    [0., 1., -1.]])

max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
X_train_maxabs  # doctest +NORMALIZE_WHITESPACE^
# array([[ 0.5, -1. ,  1. ],
#  [ 1. ,  0. ,  0. ],
#  [ 0. ,  1. , -0.5]])
X_test = np.array([[-3., -1., 4.]])
X_test_maxabs = max_abs_scaler.transform(X_test)
X_test_maxabs
# array([[-1.5, -1. ,  2. ]])
# array([ 2.,  1.,  2.])


"""
5.3.1.3 缩放有离群值的数据
如果你的数据包含许多异常值，使用均值和方差缩放可能并不是一个很好的选择。这种情况下，你可以使用 robust_scale 以及 RobustScaler 作为替代品。
它们对你的数据的中心和范围使用更有鲁棒性的估计。
"""

"""
5.3.2 非线性转换
有两种类型的转换是可用的:分位数转换和幂函数转换。分位数和幂变换都基于特征的单调变换，从而保持了每个特征值的秩。
"""

# 5.3.2.1 映射到均匀分布
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)
print(np.percentile(X_train[:, 0], [0, 25, 50, 75, 100]))
# array([ 4.3,  5.1,  5.8,  6.5,  7.9])

print(np.percentile(X_train_trans[:, 0], [0, 25, 50, 75, 100]))
# array([ 0.00... ,  0.24...,  0.49...,  0.73...,  0.99... ])

np.percentile(X_test[:, 0], [0, 25, 50, 75, 100])
# array([ 4.4  ,  5.125,  5.75 ,  6.175,  7.3  ])
np.percentile(X_test_trans[:, 0], [0, 25, 50, 75, 100])
# array([ 0.01...,  0.25...,  0.46...,  0.60... ,  0.94...])


# 5.3.2.2 映射到高斯分布
# 在许多建模场景中，需要数据集中的特征的正态化。幂变换是一类参数化的单调变换， 其目的是将数据从任何分布映射到尽可能接近高斯分布，以便稳定方差和最小化偏斜。

pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
X_lognormal = np.random.RandomState(616).lognormal(size=(3, 3))
X_lognormal
# array([[1.28..., 1.18..., 0.84...],
#        [0.94..., 1.60..., 0.38...],
#        [1.35..., 0.21..., 1.09...]])
pt.fit_transform(X_lognormal)
# array([[ 0.49...,  0.17..., -0.15...],
#        [-0.05...,  0.58..., -0.57...],
#        [ 0.69..., -0.84...,  0.10...]])


## 我们也可以 使用类 QuantileTransformer (通过设置 output_distribution='normal')把数据变换成一个正态分布。下面是将其应用到iris dataset上的结果:
quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
X_trans = quantile_transformer.fit_transform(X)
quantile_transformer.quantiles_
# array([[4.3, 2. , 1. , 0.1],
#        [4.4, 2.2, 1.1, 0.1],
#        [4.4, 2.2, 1.2, 0.1],
#        ...,
#        [7.7, 4.1, 6.7, 2.5],
#        [7.7, 4.2, 6.7, 2.5],
#        [7.9, 4.4, 6.9, 2.5]])


"""
5.3.3 归一化
归一化 是 缩放单个样本以具有单位范数 的过程。如果你计划使用二次形式(如点积或任何其他核函数)来量化任何样本间的相似度，则此过程将非常有用。
"""
# 函数 normalize 提供了一个快速简单的方法在类似数组的数据集上执行操作，使用 l1 或 l2 范式:

X = [[1., -1., 2.],
     [2., 0., 0.],
     [0., 1., -1.]]
X_normalized = preprocessing.normalize(X, norm='l2')

print(X_normalized)
# array([[ 0.40..., -0.40...,  0.81...],
#  [ 1.  ...,  0.  ...,  0.  ...],
#  [ 0.  ...,  0.70..., -0.70...]])


normalizer = preprocessing.Normalizer().fit(X)  # fit does nothing
normalizer
# Normalizer(copy=True, norm='l2')
# 在这之后归一化实例可以被使用在样本向量中，像任何其他转换器一样:
normalizer.transform(X)
# array([[ 0.40..., -0.40...,  0.81...],
#  [ 1.  ...,  0.  ...,  0.  ...],
#  [ 0.  ...,  0.70..., -0.70...]])

normalizer.transform([[-1., 1., 0.]])
# array([[-0.70...,  0.70...,  0.  ...]])


"""
要把标称型特征(categorical features) 转换为这样的整数编码(integer codes), 我们可以使用 OrdinalEncoder 。 
这个估计器把每一个categorical feature变换成 一个新的整数数字特征 (0 到 n_categories - 1):
这样的整数特征表示并不能在scikit-learn的估计器中直接使用，因为这样的连续输入，估计器会认为类别之间是有序的，但实际却是无序的。
"""

enc = preprocessing.OrdinalEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
# OrdinalEncoder(categories='auto', dtype=<... 'numpy.float64'>)
enc.transform([['female', 'from US', 'uses Safari']])
# array([[0., 1., 1.]])

"""
另外一种将标称型特征转换为能够被scikit-learn中模型使用的编码是one-of-K， 又称为 独热码或dummy encoding。 这种编码类型已经在类OneHotEncoder中实现。
该类把每一个具有n_categories个可能取值的categorical特征变换为长度为n_categories的二进制特征向量，里面只有一个地方是1，其余位置都是0。
"""

enc = preprocessing.OneHotEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
# OneHotEncoder(categorical_features=None, categories=None,
#        dtype=<... 'numpy.float64'>, handle_unknown='error',
#        n_values=None, sparse=True)
enc.transform([['female', 'from US', 'uses Safari'],
               ['male', 'from Europe', 'uses Safari']]).toarray()
# array([[1., 0., 0., 1., 0., 1.],
#        [0., 1., 1., 0., 0., 1.]])


enc.categories_
# [array(['female', 'male'], dtype=object), array(['from Europe', 'from US'], dtype=object), array(['uses Firefox', 'uses Safari'], dtype=object)]

# 可以使用参数categories_显式地指定这一点。我们的数据集中有两种性别、四种可能的大陆和四种web浏览器
genders = ['female', 'male']
locations = ['from Africa', 'from Asia', 'from Europe', 'from US']
browsers = ['uses Chrome', 'uses Firefox', 'uses IE', 'uses Safari']
enc = preprocessing.OneHotEncoder(categories=[genders, locations, browsers])
# Note that for there are missing categorical values for the 2nd and 3rd
# feature
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
# OneHotEncoder(categorical_features=None,
#        categories=[...], drop=None,
#        dtype=<... 'numpy.float64'>, handle_unknown='error',
#        n_values=None, sparse=True)
print(enc.transform([['male', 'from Asia', 'uses Chrome']]).toarray())
# array([[1., 0., 0., 1., 0., 0., 1., 0., 0., 0.]])


# 如果训练数据可能缺少分类特性，通常最好指定handle_unknown='ignore'，而不是像上面那样手动设置类别。当指定handle_unknown='ignore'，
# 并且在转换过程中遇到未知类别时，不会产生错误，但是为该特性生成的一热编码列将全部为零(handle_unknown='ignore'只支持一热编码):
enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
# OneHotEncoder(categorical_features=None, categories=None, drop=None,
#        dtype=<... 'numpy.float64'>, handle_unknown='ignore',
#        n_values=None, sparse=True)
enc.transform([['female', 'from Asia', 'uses Chrome']]).toarray()
# array([[1., 0., 0., 0., 0., 0.]])


"""
还可以使用drop参数将每个列编码为n_categories-1列，而不是n_categories列。此参数允许用户为要删除的每个特征指定类别。这对于避免某些分类器中输入矩阵的共线性是有用的。
例如，当使用非正则化回归(线性回归)时，这种功能是有用的，因为共线性会导致协方差矩阵是不可逆的。当这个参数不是None时，handle_unknown必须设置为error:
"""

X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
drop_enc = preprocessing.OneHotEncoder(drop='first').fit(X)
drop_enc.categories_
# [array(['female', 'male'], dtype=object), array(['from Europe', 'from US'], dtype=object), array(['uses Firefox', 'uses Safari'], dtype=object)]
drop_enc.transform(X).toarray()
# array([[1., 1., 1.],
#        [0., 0., 0.]])


"""
离散化 (Discretization) (有些时候叫 量化(quantization) 或 装箱(binning)) 提供了将连续特征划分为离散特征值的方法。 
某些具有连续特征的数据集会受益于离散化，因为 离散化可以把具有连续属性的数据集变换成只有名义属性(nominal attributes)的数据集。 
(译者注： nominal attributes 其实就是 categorical features, 可以译为 名称属性，名义属性，符号属性，离散属性 等)

One-hot 编码的离散化特征 可以使得一个模型更加的有表现力(expressive)，同时还能保留其可解释性(interpretability)。 
比如，用离散化器进行预处理可以给线性模型引入非线性。
"""

# 5.3.5.1 K-bins 离散化
X = np.array([[-3., 5., 15],
              [0., 6., 14],
              [6., 3., 11]])
est = preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode='ordinal').fit(X)

# 特征 1:[-∞,-1],[-1,2),[2,∞)
# 特征 2:[-∞,5),[5,∞)
# 特征 3:[-∞,14],[14,∞)
# 基于这些 bin 区间, X 就被变换成下面这样:

est.transform(X)
# array([[ 0., 1., 1.],
#        [ 1., 1., 1.],
#        [ 2., 0., 0.]])


"""
5.3.5.2 特征二值化
特征二值化 是 将数值特征用阈值过滤得到布尔值 的过程。这对于下游的概率型模型是有用的，它们假设输入数据是多值 伯努利分布(Bernoulli distribution) 。
"""
X = [[1., -1., 2.],
     [2., 0., 0.],
     [0., 1., -1.]]

binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing
print(binarizer)
# Binarizer(copy=True, threshold=0.0)

print(binarizer.transform(X))
# array([[ 1.,  0.,  1.],
#  [ 1.,  0.,  0.],
#  [ 0.,  1.,  0.]])

# 也可以为二值化器赋一个阈值:
binarizer = preprocessing.Binarizer(threshold=1.1)
binarizer.transform(X)
# array([[ 0.,  0.,  1.],
#  [ 1.,  0.,  0.],
#  [ 0.,  0.,  0.]])


"""
5.3.7 生成多项式特征
在机器学习中，通过增加一些输入数据的非线性特征来增加模型的复杂度通常是有效的。一个简单通用的办法是使用多项式特征，
这可以获得特征的更高维度和互相间关系的项。这在 PolynomialFeatures 中实现:
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.arange(6).reshape(3, 2)
X
# array([[0, 1],
#  [2, 3],
#  [4, 5]])
poly = PolynomialFeatures(2)
poly.fit_transform(X)
# array([[  1.,   0.,   1.,   0.,   0.,   1.],
#  [  1.,   2.,   3.,   4.,   6.,   9.],
#  [  1.,   4.,   5.,  16.,  20.,  25.]])


# 在一些情况下，只需要特征间的交互项，这可以通过设置 interaction_only=True 来得到

X = np.arange(9).reshape(3, 3)
X
# array([[0, 1, 2],
#  [3, 4, 5],
#  [6, 7, 8]])
poly = PolynomialFeatures(degree=3, interaction_only=True)
poly.fit_transform(X)
# array([[   1.,    0.,    1.,    2.,    0.,    0.,    2.,    0.],
#  [   1.,    3.,    4.,    5.,   12.,   15.,   20.,   60.],
#  [   1.,    6.,    7.,    8.,   42.,   48.,   56.,  336.]])



