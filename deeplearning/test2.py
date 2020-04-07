"""
@author:yux
@email:yuxer@qq.com
@time:2020/4/3 22:31
@description:  
"""
import torch.nn.functional as F
import torch
from torch.autograd import Variable
loss_fn = torch.nn.BCELoss(reduction='none',weight=torch.tensor([2.,3.,3.]))
input = torch.tensor([2., 3.3,4.])
target = torch.tensor([1., 0.,5.])
loss = loss_fn(F.sigmoid(input), target)
print(input)
print(target)
print(loss)
print(loss.mean())

