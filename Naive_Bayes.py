import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB

x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

clf = GaussianNB()
#clf = MultinomialNB() 多项式算法不能用于判断纯数字，且输入必须为非负
clf.fit(x, y)

print("result of predict:")
print(clf.predict([[-0.8, -1]]))
print("Probability of predict:")
print(clf.predict_proba([[-0.8, -1]]))
