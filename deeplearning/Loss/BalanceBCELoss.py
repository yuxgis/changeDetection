"""
@author:yux
@email:yuxer@qq.com
@time:2020/4/2 21:21
@description: 平衡BCE损失
"""
import torch.nn as nn
import torch
class BalanceBCELoss(nn.Module):
    def __init__(self):
        super(BalanceBCELoss, self).__init__()
    def forward(self,predict,target):
        #计算权重
        # 计算target 的变化与非变化比例
        pix_weight = torch.rand_like(target)
        for i, batch in enumerate(target):
            zero_count = torch.sum(batch == 0)
            one_count = torch.sum(batch == 1)
            zero_weight = zero_count.float() / (zero_count + one_count)
            one_weight = one_count.float() / (zero_count + one_count)
            pix_weight[i][batch == 0.] = zero_weight
            pix_weight[i][batch == 1.] = one_weight
        loss = nn.BCELoss(weight=pix_weight)(predict,target)
        return loss
