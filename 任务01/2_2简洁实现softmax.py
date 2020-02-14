# -*- coding: utf-8 -*-
import torch
import torchvision
import numpy as np
import sys#为了嘉爱要用d2lzh_pytorch
sys.path.append("C:/sers/dell/动手学习深度学习")
import d2lzh_pytorch as d2l
from torch import nn
from torch.nn import init

print(torch.__version__)
print(torchvision.__version__)
'''
初始化参数和获取数据
'''
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root='data')

'''
定义网络模型
'''
num_inputs = 784#28*28 784个 输入特征
num_outputs = 10

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, x): # x 的形状: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y
    
#net = LinearNet(num_inputs, num_outputs)

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x 的形状: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

from collections import OrderedDict
net = nn.Sequential(
        # FlattenLayer(),
        # LinearNet(num_inputs, num_outputs) 
        OrderedDict([
           ('flatten', FlattenLayer()),
           ('linear', nn.Linear(num_inputs, num_outputs))]) # 或者写成我们自己定义的 LinearNet(num_inputs, num_outputs) 也可以
        )
'''
初始化模型参数 初始化权重和偏差
'''
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

'''
定义损失函数
'''
loss = nn.CrossEntropyLoss() # 下面是他的函数原型
# class torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

'''
定义优化函数
'''
optimizer = torch.optim.SGD(net.parameters(), lr=0.1) # 下面是函数原型
# class torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)
'''
训练
'''
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

