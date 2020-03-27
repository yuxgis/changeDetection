import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import  KMeans
import cv2
import matplotlib.pyplot as pyplot
class PCA_CD:
    def __init__(self, after, before):
        self.after = after
        self.before = before
    def process(self):
        diff = self.after - self.before
        [w,h,d] = diff.shape
        diff = diff.reshape((w*h,d))
        print(diff)
        pca = PCA(n_components=0.75)
        data = pca.fit_transform(diff)
        k_data = KMeans(n_clusters=2, random_state=9).fit_predict(data)
        print(k_data)
        print(data)
        img = np.reshape(k_data, (w, h,))

        pyplot.imshow(np.uint8(img))
        pyplot.show()
        # scipy.misc.imsave('c.jpg', img)



import gdal
if __name__ == "__main__":
    # after = cv2.imread("../../data/abudhabi_8_after.png")
    # before = cv2.imread("../../data/abudhabi_8_before.png")
    dataset_after = gdal.Open(r"F:\变化检测数据\A1_clip.tif")
    im_width = dataset_after.RasterXSize  # 栅格矩阵的列数
    im_height = dataset_after.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset_after.RasterCount  # 波段数
    after = np.transpose(dataset_after.ReadAsArray(0, 0, im_width, im_height),(1,2,0))[0:5558,0:5314]  # 获取数据

    dataset_before = gdal.Open(r"F:\变化检测数据\B1_clip.tif")
    im_width = dataset_before.RasterXSize  # 栅格矩阵的列数
    im_height = dataset_before.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset_before.RasterCount  # 波段数
    before = np.transpose(dataset_before.ReadAsArray(0, 0, im_width, im_height),(1,2,0))[0:5558,0:5314]  # 获取数据


    pca = PCA_CD(after, before)
    pca.process()


