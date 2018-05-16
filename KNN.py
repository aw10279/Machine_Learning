import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.datasets.samples_generator import make_classification
from matplotlib.colors import ListedColormap

#生成训练集：3个分类（每类1个簇），2个特征，1000个样本
x, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, n_classes=3)
plt.scatter(x[:, 0],x[:, 1], marker="o", c=y)
plt.show()

#实例化分类器，并用训练集进行训练（K近邻并没有实际的训练过程，每次都是直接用测试集和训练集进行计算）
clf = neighbors.KNeighborsClassifier(n_neighbors=15, weights="distance")
clf.fit(x, y)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])   #自定义颜色

#确定测试集的范围
x_min, x_max = x[:, 0].min()-1, x[:, 0].max()+1
y_min, y_max = x[:, 1].min()-1, x[:, 1].max()+1

#用meshgrid生成间隔为0.02的网格点，返回的是2个矩阵，xx和yy分别代表所有网格点的2个坐标值
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

#进行预测，返回标签。ravel将矩阵降维（铺开），c_将2个一维数组按列合并，变成2个坐标一组的形式，即待预测坐标
z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

z = z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, z, cmap=cmap_light)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = 15, weights = 'distance')" )
plt.show()