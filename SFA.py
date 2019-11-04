# 慢特征图像变化检测
import numpy as np
import cv2
from scipy.linalg import inv, sqrtm, eig
import matplotlib.pyplot as pyplot


class SFA:
    def __init__(self, after, before):
        self.after = after
        self.before = before
        self.shape = None

    def process(self):
        after = self.after
        before = self.before

        self.shape = after.shape
        (rows, cols, bands) = self.shape

        # 归一化处理

        after = np.transpose(np.reshape(after, (rows * cols, bands)), (1, 0))
        before = np.transpose(np.reshape(before, (rows * cols, bands)), (1, 0))

        # 执行标准化程序
        after = self.standardization(after)
        before = self.standardization(before)
        e = np.cov(after - before)
        sum1 = np.sum((after[0] - before[0]) @ (after[0] - before[0]).T) / (rows * cols)
        sum2 = np.sum((after[1] - before[1]) @ (after[1] - before[1]).T) / (rows * cols)
        print(after)
        e_x = np.cov(after)
        e_y = np.cov(before)
        B = 1 / 2 * (e_x + e_y)
        # 特征值与特征向量
        print(B)

        (value, vector) = eig(np.linalg.inv(B) @ e)
        SFA = (vector @ after) - (vector @ before)
        print(SFA)
        tr_sfa = np.transpose(SFA, (1, 0))
        from sklearn.cluster import KMeans

        # re = np.reshape(T, (j, 1))
        kmeans = KMeans(n_clusters=2, random_state=0).fit(tr_sfa[:,1])
        img = np.reshape(kmeans.labels_, (rows, cols,))
        center = kmeans.cluster_centers_
        #tr_sfa = np.reshape(tr_sfa, (rows,cols,3))
        pyplot.imshow(img)
        pyplot.show()

    def standardization(self, data):
        (rows, cols, bands) = self.shape
        data_mean = np.mean(data, axis=1)
        data_var = np.var(data, axis=1)
        data_mean_repeat = np.tile(data_mean, (rows * cols, 1)).transpose(1, 0)
        data_var_repeat = np.tile(data_var, (rows * cols, 1)).transpose(1, 0)
        data = (data - data_mean_repeat) / data_var_repeat
        print(data)
        return data

import gdal
if __name__ == "__main__":
    after = cv2.imread("../../data/abudhabi_8_after.png")
    before = cv2.imread("../../data/abudhabi_8_before.png")

    # dataset_after = gdal.Open(r"F:\deeplearndata\rssrai2019_change_detection\train\train\img_2017\image_2017_960_960_1.tif")
    # im_width = dataset_after.RasterXSize  # 栅格矩阵的列数
    # im_height = dataset_after.RasterYSize  # 栅格矩阵的行数
    # im_bands = dataset_after.RasterCount  # 波段数
    # after = np.transpose(dataset_after.ReadAsArray(0, 0, im_width, im_height),(1,2,0))  # 获取数据
    #
    # dataset_before = gdal.Open(r"F:\deeplearndata\rssrai2019_change_detection\train\train\img_2018\image_2018_960_960_1.tif")
    # im_width = dataset_before.RasterXSize  # 栅格矩阵的列数
    # im_height = dataset_before.RasterYSize  # 栅格矩阵的行数
    # im_bands = dataset_before.RasterCount  # 波段数
    # before = np.transpose(dataset_before.ReadAsArray(0, 0, im_width, im_height),(1,2,0))  # 获取数据
    sfa = SFA(after, before)
    sfa.process()
