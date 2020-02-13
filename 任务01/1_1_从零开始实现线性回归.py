# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
'''
1.生成数据集
'''
# set input feature number 
#设置输入特征的数量；在这里用了两个特征，所以设为2
num_inputs = 2
# set example number
num_examples = 1000

# set true weight and bias in order to generate corresponded label
#设置真实的权重和偏差，这两个参数也是要训练学习的两个参数，在这里设置是为了通过特诊来生成对应的标签
true_w = [2, -3.4]
true_b = 4.2

features = torch.randn(num_examples, num_inputs,
                      dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
#实际上不可能直接符合这个偏差，因此又再加上了一个偏差，是呈正态分布的偏差，随机生成的
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)
'''
2.读取数据集
'''
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))# 列表
    random.shuffle(indices)  # random read 10 samples 打乱数据，而不是按照顺序排列的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # the last time may be not enough for a whole batch
        #考虑最后一组可能批量不足
        yield  features.index_select(0, j), labels.index_select(0, j)
batch_size = 10
'''
3.初始化模型参数
'''
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
#附加梯度，才能通过链式法则，反向传播求梯度
'''
4.定义模型
'''
def linreg(X, w, b):
    return torch.mm(X, w) + b
#输出预测值
    
'''
5、定义损失函数-采用均方误差
'''
def squared_loss(y_hat, y): 
    return (y_hat - y.view(y_hat.size())) ** 2 / 2
#view函数用来将tensor y的shape和y_hat进行统一

'''
6.定义优化函数-小批量随机梯度下降
'''    
def sgd(params, lr, batch_size): 
    for param in params:
        param.data -= lr * param.grad / batch_size 
        # ues .data to operate param without gradient track
        #用.data是不想参数被附加  
'''
训练
'''   
# super parameters init
#超参数的初始化，两个超参数
lr = 0.03  #学习率
num_epochs = 5 #训练周期

net = linreg  #网络，这里是一个单层的线性网络
loss = squared_loss #损失函数，使用的是均方误差损失函数

# training
for epoch in range(num_epochs):  # training repeats num_epochs times 训练周期的循环
    # in each epoch, all the samples in dataset will be used once
    #每个训练周期，全部数据都使用一次
    
    # X is the feature and y is the label of a batch sample
    for X, y in data_iter(batch_size, features, labels):
        #print(X,'\n',y)
        l = loss(net(X, w, b), y).sum()  
        # calculate the gradient of batch sample loss 
        #计算出的损失   net(X,w,b)是预测值
        l.backward() #反向传播求梯度 
        # using small batch random gradient descent to iter model parameters
        sgd([w, b], lr, batch_size) 
        #sgd(需要优化的参数, 学习率， 批量大小）
        
        #参数梯度的清零，因为后面会累加，不清零会影响结果
        # reset parameter gradient
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)#计算损失
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))#周期数   和  周期对应损失     
        
