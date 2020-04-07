"""
@author:yux
@email:yuxer@qq.com
@time:2020/4/6 20:19
@description:  最终损失
"""
import torch
import torch.nn as nn
from Loss.DiceLoss import DiceLoss
from Loss.BalanceBCELoss import BalanceBCELoss

class CustomLoss(nn.Module):
    def __init__(self,nmn):
        super(CustomLoss, self).__init__()
        self.namna = nmn
    def forward(self,predict_array,target):#含有batch
        final_loss = torch.zeros([1])
        layer_weight = [0.5, 0.5, 0.75, 0.5, 1.0]
        for i,predict in enumerate(predict_array):
            #分别计算损失
            predict = torch.sigmoid(predict)

            balance_loss = BalanceBCELoss()(predict,target)
            dice_loss = DiceLoss()(predict,target)*self.namna
            loss = (balance_loss+dice_loss)*layer_weight[i]

            final_loss += loss
        return final_loss


