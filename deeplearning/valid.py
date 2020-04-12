"""
@author:yux
@email:yuxer@qq.com
@time:2020/4/11 12:36
@description:  验证模型
"""
from torch.autograd import Variable
from torch.utils.data import DataLoader

from RemoteImageDataset import RemoteImageDataset
from UNet_Plus import UNet_Plus
from UpdateNestUnet import UpdateNestUnet
import torch
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def valid():
    # model = UNet_Plus(6).to(device)
    model = UpdateNestUnet(in_ch=3).to(device)
    model.load_state_dict(torch.load("./Pth/val_no_enhance_conc_adam_l2_1805.pth", map_location='cpu'))
    loader = DataLoader(RemoteImageDataset(
        r"H:\yux\data\256png\valid"), batch_size=1)
    model.eval()
    with torch.no_grad():
        for img1, img2, mask in loader:
            images1 = Variable(img1)
            images2 = Variable(img2)

            output = model(images1, images2)
            img_y = torch.squeeze(output[-1]).numpy()
            # img_y[img_y >= 0.5] = 255
            # img_y[img_y < 0.5] = 0
            plt.subplot(1, 2, 1)
            plt.imshow(img_y)
            plt.subplot(1, 2, 2)
            plt.imshow(torch.squeeze(mask).numpy())
            plt.show()
if __name__ == "__main__":
    valid()
