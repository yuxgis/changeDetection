import glob
from PIL import Image
import numpy as np
def accuracy(true_path,predict_path):
    true_file_list = glob.glob(true_path+"/*.jpg")
    preditc_file_list = glob.glob(predict_path+"/*.png")
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for i in range(len(true_file_list)):
        t_img = np.asarray(Image.open(true_file_list[i]))
        t_img.flags.writeable = True
        t_img[t_img<=1] = 0
        t_img[t_img>1] = 255
        p_img = np.asarray(Image.open(preditc_file_list[i]))[:,:,0]
        print(np.unique(t_img,return_counts=True))
        print(np.unique(p_img,return_counts=True))
        tp = (t_img == 0) & (p_img == 68)
        fn = (t_img == 0) & (p_img == 253)
        fp = (t_img == 255) & (p_img == 68)
        tn = (t_img == 255) & (p_img == 253)
        #print(np.sum(tp)+np.sum(fn)+np.sum(fp)+np.sum(tn))
        accuracy.append((np.sum(tp)+np.sum(tn))/(np.sum(tp)+np.sum(fn)+np.sum(fp)+np.sum(tn)))
        pre = np.sum(tp)/(np.sum(tp)+np.sum(fp))
        precision.append(pre)
        rec = np.sum(tp)/(np.sum(tp)+np.sum(fn))
        recall.append(rec)
        f1.append(2*pre*rec/(pre+rec))
    print(np.mean(np.array(accuracy)))
    print(np.mean(np.array(precision)))
    print(np.mean(np.nan_to_num(np.array(recall))))
    print(np.mean(np.nan_to_num(np.array(f1))))
accuracy(r"H:\yux\data\ChangeDetectionDataset\ChangeDetectionDataset\Real\subset\test\OUT",r"H:\yux\data\ChangeDetectionDataset\ChangeDetectionDataset\Real\subset\test\PREDICT")

