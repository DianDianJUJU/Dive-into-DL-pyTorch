# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np
torch.manual_seed(1)

print(torch.__version__)
torch.set_default_tensor_type('torch.FloatTensor')
'''
1.生成数据集
'''
num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
'''
2.读取数据集
'''
import torch.utils.data as Data

batch_size = 10

# combine featues and labels of dataset
#数据和标签组合起来形成数据集
dataset = Data.TensorDataset(features, labels)

# put dataset into DataLoader
#从数据集中取数据，需要几个参数
data_iter = Data.DataLoader(
    dataset=dataset,            # torch TensorDataset format
    batch_size=batch_size,      # mini batch size  每次取出的批量大小
    shuffle=True,               # whether shuffle the data or not  是否混淆，随机取出
    #num_workers=2,              # read data in multithreading  工作线程=2
)

'''
3.定义模型
'''
class LinearNet(nn.Module):#线性网络
    def __init__(self, n_feature):#初始化
        super(LinearNet, self).__init__()      # call father function to init 
        self.linear = nn.Linear(n_feature, 1)  # function prototype: `torch.nn.Linear(in_features, out_features, bias=True)`
#两个输入特征，一个输出特征
    def forward(self, x):
        y = self.linear(x)#y是输出
        return y
    
net = LinearNet(num_inputs)#实例化  num_inputs=2
print(net)#单层线性网络，两个输入，一个输出，存在偏差
# ways to init a multilayer network
#初始化多层网络
# method one
#方法一：调用神经网络nn中的Sequential函数，把不同的层作为参数添加进去
#用这个方法实会初始化相关的weight和bias
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # other layers can be added here
    )

#方法二
# method two
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))#添加  add_module(他的title，是一个什么层)
# net.add_module ......

#方法三：作为参数传入， 只是把神经网络作为参数放入了字典里面
# method three
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))

print(net)#生成的全部
print(net[0])#生成的第一层
'''
4.初始化模型参数
'''
from torch.nn import init
#init中有不同的初始化方式
init.normal_(net[0].weight, mean=0.0, std=0.01)#（要初始化的变量，输入初始化的特征）
init.constant_(net[0].bias, val=0.0)  # or you can use `net[0].bias.data.fill_(0)` to modify it directly
'''
5.定义损失函数
'''
loss = nn.MSELoss()    # nn built-in squared loss function
                       # function prototype: `torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
'''
6.定义优化函数
'''        
import torch.optim as optim
#                    （要优化的函数   ， 超参数学习率）
optimizer = optim.SGD(net.parameters(), lr=0.03)   # built-in random gradient descent function
print(optimizer)  # function prototype: `torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)`
'''
7.训练
''' 
num_epochs = 3#训练周期
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)#将输入X预测
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # reset gradient, equal to net.zero_grad()
        l.backward()
        optimizer.step()#优化函数迭代优化
    print('epoch %d, loss: %f' % (epoch, l.item()))
'''
8.结果
'''   
# result comparision
dense = net[0]
print(true_w, dense.weight.data)
print(true_b, dense.bias.data)        