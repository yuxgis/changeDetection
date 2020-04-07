import numpy as np
import torch.nn as nn
import torch
from torchvision import models
import torchvision
class Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(Block,self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out
class UNet_Plus(nn.Module):
    def __init__(self,input_channel=6):
        super(UNet_Plus,self).__init__()
        channels = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0_0 = Block(input_channel,channels[0],channels[0])
        self.conv1_0 = Block(channels[0],channels[1],channels[1])
        self.conv2_0 = Block(channels[1],channels[2],channels[2])
        self.conv3_0 = Block(channels[2],channels[3],channels[3])
        self.conv4_0 = Block(channels[3],channels[4],channels[4])

        self.conv0_1 = Block(channels[0]+channels[1], channels[0], channels[0])
        self.conv1_1 = Block(channels[1]+channels[2], channels[1], channels[1])
        self.conv2_1 = Block(channels[2]+channels[3], channels[2],channels[2])
        self.conv3_1 = Block(channels[3]+channels[4], channels[3], channels[3])

        self.conv0_2 = Block(channels[0]*2+channels[1],channels[0],channels[0])
        self.conv1_2 = Block(channels[1]*2+channels[2],channels[1],channels[1])
        self.conv2_2 = Block(channels[2]*2+channels[3],channels[2],channels[2])

        self.conv0_3 = Block(channels[0]*3+channels[1],channels[0],channels[0])
        self.conv1_3 = Block(channels[1]*3+channels[2],channels[1],channels[1])

        self.conv0_4 = Block(channels[0]*4+channels[1],channels[0],channels[0])
        self.final = nn.Conv2d(channels[0],1,kernel_size=1)


    def forward(self,before,after): #输入不同时相的图像
        input = torch.cat([before,after],1)
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))


        output0_4 = self.final(x0_4)
        output0_3 = self.final(x0_3)
        output0_2 = self.final(x0_2)
        output0_1 = self.final(x0_1)
        output0_0 = self.final(x0_0)

        return output0_0, output0_1, output0_2, output0_3, output0_4




