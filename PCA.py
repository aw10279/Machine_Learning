import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

x, y = make_blobs(n_samples=10000, n_features=3, centers=[[3,3,3], [4,4,4], [1,1,1], [2,2,2]], cluster_std=[0.2, 0.1, 0.2, 0.1], 
                  random_state=9)
#10000个样本，3个特征，聚集成4个簇
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=30)
#rect目测为缩放功能，elev和azim为角度调整
plt.scatter(x[:, 0], x[:, 1], x[:, 2],marker='o')
plt.show()

pca = PCA(n_components = 2) #实例化
pca.fit(x)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

x_new = pca.transform(x) #将样本投影到降维后的空间，形成新样本集
plt.scatter(x[:, 0], x[:, 1],marker='o')
plt.show()

pca_auto = PCA(n_components=0.95) 
#不指定维数，只规定降维后的方差比例，0.95只能保留第一维，若0.99将保留前两维
#若用PCA(n_components='mle')让算法自动选择，也只会保留第一维

pca_auto.fit(x)
print (pca_auto.explained_variance_ratio_)
print (pca_auto.explained_variance_)
print (pca_auto.n_components_)
