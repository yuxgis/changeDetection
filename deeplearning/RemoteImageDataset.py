"""
@author:yux
@email:yuxer@qq.com
@time:2020/4/2 14:25
@description:  遥感影像数据加载
"""

from torch.utils.data import Dataset
import glob
import torch
from PIL import Image

from torchvision import transforms
#收集文件名


#读取GAN数据集
class CollectFile():
    def __init__(self,path):
        self.before_file_list = glob.glob(path+"/A/*.jpg")
        self.after_file_list = glob.glob(path+"/B/*.jpg")
        self.result_file_list = glob.glob(path+"/OUT/*.jpg")

    def __getitem__(self,index):
        return (self.before_file_list[index],self.after_file_list[index],self.result_file_list[index])

    def __len__(self):
        return len(self.result_file_list)

#OSCD数据集加载
class OscdFile():
    def __init__(self,root):
        self.before_file_list = glob.glob(root + r"/*before.png")
        self.after_file_list = glob.glob(root + r"/*after.png")
        self.mask_file_list = glob.glob(root + r"/*mask.png")

    def __getitem__(self, index):
        return (self.before_file_list[index], self.after_file_list[index], self.mask_file_list[index])

    def __len__(self):
        return len(self.mask_file_list)

class RemoteImageDataset(Dataset):
    def __init__(self,path):
        super(RemoteImageDataset).__init__()
        #self.file_list = CollectFile(path)
        self.file_list = OscdFile(path)
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self,index):
        #print(index)
        image_arrays = torch.FloatTensor(1,3).zero_()


        #image_arrays[i] = Image.open(data[index])
        #print(self.file_list[index])
        before = transforms.ToTensor()(Image.open(self.file_list[index][0]))
        after = transforms.ToTensor()(Image.open(self.file_list[index][1]))
        change = transforms.ToTensor()(Image.open(self.file_list[index][2]))
        # n = before.numpy()
        # c =  change.numpy()

        return before, after, change

from torch.utils.data import DataLoader
if __name__ == "__main__":
    loader = DataLoader(RemoteImageDataset(r"E:\迅雷下载\ChangeDetectionDataset\ChangeDetectionDataset\Real\subset\train"),batch_size=4)
    #print(iter(loader))

    for a,b,c in loader:
        print(a)








