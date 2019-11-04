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




if __name__ == "__main__":
    after = cv2.imread("../../data/abudhabi_8_after.png")
    before = cv2.imread("../../data/abudhabi_8_before.png")
    pca = PCA_CD(after, before)
    pca.process()


