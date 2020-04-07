"""
@author:yux
@email:yuxer@qq.com
@time:2020/4/2 13:45
@description: 训练文件
"""
import torch
from RemoteImageDataset import RemoteImageDataset
from torch.utils.data import DataLoader
from UNet_Plus import UNet_Plus
from Loss.CustomLoss import CustomLoss
from torch import autograd, optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def train(epochs):
    model = UNet_Plus(6).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01)
    for i in range(epochs):

        epoch_loss = 0
        loader = DataLoader(RemoteImageDataset(
            r"E:\迅雷下载\ChangeDetectionDataset\ChangeDetectionDataset\Real\subset\train"),batch_size=1)
        for a,b,c in loader:
            optimizer.zero_grad()
            m = model(a,b)
            loss = CustomLoss(nmn=0.5)(m,c)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(loss.item())
        print(epoch_loss)






if __name__=="__main__":
    train(50)


