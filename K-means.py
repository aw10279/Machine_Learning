import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics #聚类效果评估工具
#%matplotlib inline   在Jupiter中将图嵌入文档，而非跳窗显示
x, y = make_blobs(n_samples=1000, n_features=4, centers=[[-2,-2], [0,0], [2,2],[5,5]], cluster_std=[0.4,0.5,0.3,0.4], random_state=15)
#random_state用法同random seed，即随机种子，用来制约随机数的生成。只要该参数固定，生成的随机数会完全一样。
plt.scatter(x[:,0], x[:,1],marker="o")
#x[:,0]就是取所有行的第0个数据, x[:,1] 就是取所有行的第1个数据,是numpy多维数组的一种写法。
plt.show()

y_pred = KMeans(n_clusters=4, random_state=15).fit_predict(x)
plt.scatter(x[:,0], x[:,1],c=y_pred)
plt.show()
evalue = metrics.calinski_harabaz_score(x, y_pred)
print(evalue)