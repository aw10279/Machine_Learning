import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

x1, y1 = datasets.make_circles(n_samples=5000, factor=.6, noise=.05)
#factor控制内外环的间距，=1时变为一整条环
x2, y2 = datasets.make_blobs(n_samples=1000, n_features=2,centers=[[1.2,1.2]], cluster_std=[[.1]],random_state=9)
X = np.concatenate((x1, x2))  #按列合并两个数组（默认）

plt.scatter(X[:, 0], X[:, 1], marker="o")
#plt.show()

y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)  #KMeans表现欠佳，对比用
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

y_pred = DBSCAN(eps=0.1, min_samples=10).fit_predict(X)
#此函数的关键是eps和samples。eps和min_samples为默认值时无分类，eps=0.1时只有2个分类，同时让min_samples=10才出现正确的3个分类。
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

'''eps过大，则更多的点会落在核心对象的邻域，此时我们的类别数可能会减少， 本来不应该是一类的样本也会被划为一类。
反之则类别数可能会增大，本来是一类的样本却被划分开。

在eps一定的情况下，min_samples过大，则核心对象会过少，此时簇内部分本来是一类的样本可能会被标为噪音点，类别数也会变多。
反之min_samples过小的话，则会产生大量的核心对象，可能会导致类别数过少。'''

