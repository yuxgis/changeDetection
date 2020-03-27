import numpy as np
from scipy.linalg import inv, sqrtm, eig
import matplotlib.pyplot as pyplot
import numpy.matlib as nplib
class MAD:
    def __init__(self,after,before):
        print(print(np.__version__))
        self.after = after
        self.before = before
    def propess(self):
        #进行相应的处理计算
        # dataset_after = gdal.Open(
        #     r"F:\deeplearndata\rssrai2019_change_detection\train\train\img_2017\image_2017_960_960_8.tif")
        # im_width = dataset_after.RasterXSize  # 栅格矩阵的列数
        # im_height = dataset_after.RasterYSize  # 栅格矩阵的行数
        # im_bands = dataset_after.RasterCount  # 波段数
        # after = np.transpose(dataset_after.ReadAsArray(0, 0, im_width, im_height), (1, 2, 0))  # 获取数据
        #
        # dataset_before = gdal.Open(
        #     r"F:\deeplearndata\rssrai2019_change_detection\train\train\img_2018\image_2018_960_960_8.tif")
        # im_width = dataset_before.RasterXSize  # 栅格矩阵的列数
        # im_height = dataset_before.RasterYSize  # 栅格矩阵的行数
        # im_bands = dataset_before.RasterCount  # 波段数
        # before = np.transpose(dataset_before.ReadAsArray(0, 0, im_width, im_height), (1, 2, 0))  # 获取数据
        after = self.after
        before = self.before
        (rows, cols, bands) = after.shape

        # 归一化处理

        after = np.transpose(np.reshape(after, (rows * cols, bands)), (1, 0))
        before = np.transpose(np.reshape(before, (rows * cols, bands)), (1, 0))

        after_mean = np.mean(after, axis=1)
        after_var = np.std(after, axis=1)
        before_mean = np.mean(before, axis=1)
        before_var = np.std(before, axis=1)

        for i in range(bands):
            #test = after[:, i] - after_mean[i]
            after[i,:] = (after[i,:]-after_mean[i])/after_var[i]
            before[i,:] = (before[i,:]-before_mean[i])/before_var[i]

        cov_aa_mari = np.cov(after)
        cov_aa_mat_i = np.linalg.inv(cov_aa_mari)
        con_cov = np.cov(after, before)
        cov_xx = con_cov[0:bands, 0:bands]
        cov_xy = con_cov[0:bands, bands:]
        cov_yx = con_cov[bands:, 0:bands]
        cov_yy = con_cov[bands:, bands:]
        # yy_cov = np.cov(before)
        A = inv(cov_xx) @ cov_xy @ inv(cov_yy) @ cov_yx
        B = inv(cov_yy) @ cov_yx @ inv(cov_xx) @ cov_xy  # 与A特征值相同，但特征向量不同

        # A的特征值与特征向量 av 特征值， ad 特征向量
        [av, ad] = eig(A)

        # 对特征值从小到大排列 与 CCA相反
        swap_av_index = np.argsort(av)
        swap_av = av[swap_av_index[:av.size:1]]
        swap_ad = ad[swap_av_index[:av.size:1], :]

        # 满足st 条件
        ma = inv(sqrtm(swap_ad.T @ cov_xx @ swap_ad))  # 条件一

        swap_ad = swap_ad @ ma

        # 对应b的值
        [bv, bd] = eig(B)
        swap_bv = bv[swap_av_index[:bv.size:1]]
        swap_bd = bd[swap_av_index[:bd.size:1]]
        mb = inv(sqrtm(swap_bd.T @ cov_yy @ swap_bd))  # 条件二

        swap_bd = swap_bd @ mb
        # ab = np.linalg.inv(cov_yy) @ cov_yx @ swap_ad
        # bb = np.linalg.inv()

        MAD = swap_ad.T @ after - (swap_bd.T @ before)
        [i, j] = MAD.shape
        var_mad = np.zeros(i)
        for k in range(i):
            var_mad[k] = np.var(MAD[k])

        var_mad = np.transpose(nplib.repmat(var_mad, j, 1), (1, 0))
        res = MAD * MAD / var_mad
        T = res.sum(axis=0)
        # T = np.zeros(j)
        # #for row in range(j):
        # sum = 0.
        # for col in range(i):
        #     sum = np.sum(np.square(MAD[col,:] / np.var(MAD[col])))
        #     T[i] = sum
        # Kmeans 聚类
        from sklearn.cluster import KMeans

        re = np.reshape(T, (j, 1))
        kmeans = KMeans(n_clusters=2, random_state=0).fit(re)
        img = np.reshape(kmeans.labels_, (rows, cols,))
        center = kmeans.cluster_centers_
        pyplot.imshow(np.uint8(img))
        pyplot.show()
        # scipy.misc.imsave('c.jpg', img)
        print(center)
import gdal
if __name__=="__main__":
    dataset_after = gdal.Open(r"F:\变化检测数据\A1_clip.tif")
    im_width = dataset_after.RasterXSize  # 栅格矩阵的列数
    im_height = dataset_after.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset_after.RasterCount  # 波段数
    after = np.transpose(dataset_after.ReadAsArray(0, 0, im_width, im_height), (1, 2, 0))[0:5558, 0:5314]  # 获取数据

    dataset_before = gdal.Open(r"F:\变化检测数据\B1_clip.tif")
    im_width = dataset_before.RasterXSize  # 栅格矩阵的列数
    im_height = dataset_before.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset_before.RasterCount  # 波段数
    before = np.transpose(dataset_before.ReadAsArray(0, 0, im_width, im_height), (1, 2, 0))[0:5558, 0:5314]  # 获取数据
    mad = MAD(after,before)
    mad.propess()