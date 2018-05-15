import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics #聚类效果评估工具

data, lable = make_blobs(n_samples=1000, n_features=2, centers=[[-2,-2], [0,0], [2,2],[5,5]], cluster_std=[0.4,0.5,0.3,0.4], random_state=15)
#random_state用法同random seed，即随机种子，用来制约随机数的生成。只要该参数固定，生成的随机数会完全一样。
#x保存数据集，y保存标签（分成几类就有几种标签）
#因需要数据可视化，n_features只取2（二维数据），实际可处理多维数据，同时centers必须写多维坐标或只写个数（写二维坐标就默认为二维数据）

plt.scatter(data[:,0], data[:,1],marker="o")
#x[:,0]就是取所有行的第0个数据, x[:,1] 就是取所有行的第1个数据,是numpy多维数组的一种写法。
plt.show()

#用Kmeans处理数据集x，返回值为所有样本的标签。
lable_pred = KMeans(n_clusters=4, random_state=15).fit_predict(data)
#fit_predict是fit和lable(或predict)两个函数的合并，即聚类的同时获取标签。

data0 = data[lable_pred==0]
data1 = data[lable_pred==1]
data2 = data[lable_pred==2]
#可将分类后每组的样本分类汇总

plt.scatter(data[:,0], data[:,1],c=lable_pred)
plt.show()

#计算此次聚类的质量得分
evalue = metrics.calinski_harabaz_score(data, lable_pred)
print(evalue)