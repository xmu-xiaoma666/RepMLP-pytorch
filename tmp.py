import torch
from torch import nn
from collections import OrderedDict
from torch.nn import functional as F
import numpy as np
from numpy import random

c=8
h=7
w=7
o=18
bs=128

input=torch.randn(bs,c,h,w)
input2=input.reshape(bs,c*h*w)
input3=input.reshape(bs,c*h*w,1,1)
#### conv
kernel=torch.randn(o,c,h,w)
conv_out=F.conv2d(input,weight=kernel,padding=h//2,stride=1)



I=torch.eye(c*h*w).reshape(-1,c,h,w)

# fc_weight=F.conv2d(I,weight=kernel,padding=h//2,stride=1)

fc_conv=nn.Conv2d(c,o,1,padding=h//2,stride=1,bias=False)
fc_conv.weight.data=kernel
fc_weight=fc_conv(I)

fc_weight=fc_weight.reshape(c*h*w,o*h*w)
fc_weight2=fc_weight.t().reshape(o*h*w,c*h*w,1,1)
fc_out1=torch.matmul(input2,fc_weight)
fc_out2=F.conv2d(input3,weight=fc_weight2,stride=1).reshape(bs,-1)

conv_out=conv_out.reshape(-1,o*h*w)
print(conv_out.shape)
print(fc_out2.shape)
print(((conv_out-fc_out1)**2).sum())
print(((conv_out-fc_out2)**2).sum())
# 
