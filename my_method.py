import numpy as np
import cv2
from sklearn import datasets
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import matplotlib.pyplot as pyplot
import gdal
#X, y = datasets.make_blobs(n_samples=500, n_features=6, centers=5, cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4], random_state=11)
after = cv2.imread("../../data/abudhabi_8_after.png")
before = cv2.imread("../../data/abudhabi_8_before.png")
temp = after - before
# dataset_after = gdal.Open(r"F:\deeplearndata\rssrai2019_change_detection\train\train\img_2017\image_2017_960_960_8.tif")
# im_width = dataset_after.RasterXSize  # 栅格矩阵的列数
# im_height = dataset_after.RasterYSize  # 栅格矩阵的行数
# im_bands = dataset_after.RasterCount  # 波段数
# after = np.transpose(dataset_after.ReadAsArray(0, 0, im_width, im_height),(1,2,0))  # 获取数据
#
# dataset_before = gdal.Open(r"F:\deeplearndata\rssrai2019_change_detection\train\train\img_2018\image_2018_960_960_8.tif")
# im_width = dataset_before.RasterXSize  # 栅格矩阵的列数
# im_height = dataset_before.RasterYSize  # 栅格矩阵的行数
# im_bands = dataset_before.RasterCount  # 波段数
# before = np.transpose(dataset_before.ReadAsArray(0, 0, im_width, im_height),(1,2,0))  # 获取数据
# (rows , cols, bands) = (im_width, im_height, im_bands)
#图像值归一化处理

(rows , cols, bands) = after.shape
after = np.transpose(np.reshape(after, (rows * cols, bands)), (1, 0))/255
before = np.transpose(np.reshape(before, (rows * cols, bands)), (1, 0))/255
con_cov = np.cov(after[0:3], before[0:3])
print(con_cov)

corr = con_cov
s_corr = np.array([corr[i,i] for i in range(bands)])
s_corr = s_corr/np.sum(s_corr)
diff_img = after - before
result_img = np.zeros((bands,rows*cols),np.float32)
for i in range(bands):
    result_img[i] = diff_img[i] * s_corr[i]
from sklearn.cluster import KMeans
tr = np.transpose(diff_img,(1,0))
kmeans = KMeans(n_clusters=2, random_state=0).fit(tr)
img = np.reshape(tr, (rows, cols,3))

pyplot.imshow(np.uint8(temp))
pyplot.show()




