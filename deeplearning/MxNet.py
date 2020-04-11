# @Time    : 2019/12/12 12:24
# @Author  : yux
# @Content : 混合Unet模型

import torch.nn as nn
import torch
import numpy as np
from torch import autograd


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            #nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            #nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class MxUnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MxUnet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)

        self.catconv = DoubleConv(2, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1536, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(256*3, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(128*3, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(64*3, 64)

        #self.conv10 = nn.Conv2d(64,out_ch, 1)

        self.allmask = nn.Sequential(
            nn.Conv2d(128,out_ch,1),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid()
                      )
        #self.allmask = nn.Conv2d(128,out_ch,1)

    def forward(self, x1, x2):
        # before 以b开头
        bconv1 = self.conv1(x1)
        bpool1 = nn.MaxPool2d(2)(bconv1)
        bconv2 = self.conv2(bpool1)
        bpool2 = nn.MaxPool2d(2)(bconv2)
        bconv3 = self.conv3(bpool2)
        bpool3 = nn.MaxPool2d(2)(bconv3)
        bconv4 = self.conv4(bpool3)
        bpool4 = nn.MaxPool2d(2)(bconv4)
        bconv5 = self.conv5(bpool4)
        bup1 = self.up6(bconv5)

        # after 以a开头
        aconv1 = self.conv1(x2)
        apool1 = nn.MaxPool2d(2)(aconv1)
        aconv2 = self.conv2(apool1)
        apool2 = nn.MaxPool2d(2)(aconv2)
        aconv3 = self.conv3(apool2)
        apool3 = nn.MaxPool2d(2)(aconv3)
        aconv4 = self.conv4(apool3)
        apool4 = nn.MaxPool2d(2)(aconv4)
        aconv5 = self.conv5(apool4)
        aup1 = self.up6(aconv5)

        # before 以b开头
        bconcat1 = torch.cat([bup1, bconv4, aconv4], dim=1)
        bconv6 = self.conv6(bconcat1)
        bup2 = self.up7(bconv6)
        bconcat2 = torch.cat([bup2,bconv3,aconv3],dim=1)
        bconv7 = self.conv7(bconcat2)
        bup3 = self.up8(bconv7)
        bconcat3 = torch.cat([bup3,bconv2,aconv2],dim=1)
        bconv8 = self.conv8(bconcat3)
        bup4 = self.up9(bconv8)
        bconcat4 = torch.cat([bup4,bconv1,aconv1],dim = 1)
        bconv9 = self.conv9(bconcat4)

        # after 以a开头
        aconcat1 = torch.cat([aup1, aconv4, bconv4], dim=1)
        aconv6 = self.conv6(aconcat1)
        aup2 = self.up7(aconv6)
        aconcat2 = torch.cat([aup2, aconv3, bconv3], dim=1)
        aconv7 = self.conv7(aconcat2)
        aup3 = self.up8(aconv7)
        aconcat3 = torch.cat([aup3, aconv2, bconv2], dim=1)
        aconv8 = self.conv8(aconcat3)
        aup4 = self.up9(aconv8)
        aconcat4 = torch.cat([aup4, aconv1, bconv1], dim=1)
        aconv9 = self.conv9(aconcat4)


        # before after 合并

        dcat1 = torch.cat([bconv9,aconv9],dim=1)
        result = self.allmask(dcat1)
        result = nn.Sigmoid()(result)
        #print(result.size())
        return result







