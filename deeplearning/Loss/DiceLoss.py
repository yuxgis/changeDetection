"""
@author:yux
@email:yuxer@qq.com
@time:2020/4/3 19:35
@description:  Dice Loss 损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self,predict,target):
        num = predict.size(0)
        smooth = 1
        weight = 0.5
        probs = torch.sigmoid(predict)
        p = probs.view(num,-1) #flat 操作
        t = target.view(num,-1)
        intersection = p * t
        score = 2. * (intersection.sum(1)+smooth)/(p.sum(1)+weight*t.sum(1) + smooth)
        return score

