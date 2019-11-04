#谱聚类
import numpy as np
import cv2
from sklearn import datasets
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import matplotlib.pyplot as pyplot
#X, y = datasets.make_blobs(n_samples=500, n_features=6, centers=5, cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4], random_state=11)
after = cv2.imread("../../data/abudhabi_5_after.png")
before = cv2.imread("../../data/abudhabi_5_before.png")
X =  after - before
X = X[80:160,80:160,:]
(rows,cols,bands ) = X.shape
X= np.reshape(X,(rows*cols,bands))[0:4900,:]
y_pred = cv2.imread("../../data/abudhabi_5_mask.png")[:,:,0]
y_pred = y_pred[80:160,80:160]

y_pred = np.reshape(y_pred,(rows*cols))[0:4900]
y_pred[y_pred == 255] = 1

for index, gamma in enumerate((0.1,1,10)):
    y_pred = SpectralClustering(n_clusters=2, gamma=gamma).fit_predict(X)
    img = np.reshape(y_pred, (70, 70,))
    pyplot.imshow(np.uint8(img))
    pyplot.show()
    print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", 2,"score:", metrics.calinski_harabaz_score(X, y_pred))

print(X)

