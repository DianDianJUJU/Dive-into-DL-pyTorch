# -*- coding: utf-8 -*-
import torch
import torchvision
import numpy as np
import sys#为了嘉爱要用d2lzh_pytorch
sys.path.append("C:/sers/dell/动手学习深度学习")
import d2lzh_pytorch as d2l

print(torch.__version__)
print(torchvision.__version__)

'''
1.获取训练集数据和测试集数据
'''
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root='data')

'''
2.模型参数初始化
'''
num_inputs = 784
print(28*28)
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
#梯度
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
'''
3.对多维Tensor按维度操作
'''
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))  # dim为0，按照相同的列求和，并在结果中保留列特征
print(X.sum(dim=1, keepdim=True))  # dim为1，按照相同的行求和，并在结果中保留行特征
print(X.sum(dim=0, keepdim=False)) # dim为0，按照相同的列求和，不在结果中保留列特征
print(X.sum(dim=1, keepdim=False)) # dim为1，按照相同的行求和，不在结果中保留行特征

'''
4.定义softmax操作
'''
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    # print("X size is ", X_exp.size())
    # print("partition size is ", partition, partition.size())
    return X_exp / partition  # 这里应用了广播机制

'''
5.定义softmax模型
'''
def net(X):
    print((torch.mm(X.view((-1, num_inputs)), W) + b).type())
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

'''
6.定义损失函数--交叉熵
'''
def cross_entropy(y_hat, y):#输入 y的估计值 和 真实的y
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))
#从预测值里面 取数据，取真实值-（对应one-hot编码中的1）对应的哪一个
'''
7.定义准确率
'''
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()
'''
- y_hat.**argmax**(dim=1)
取最大的数，dim=1，按照行取每行最大
取出来和真实值相同则为1，加起来就是所有正确的
再取平均，就是准确率
'''
# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
#评价模型的准确率  这个模型平均的准确率  准确率全部累加除以样本数
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n
'''
训练模型
'''
num_epochs, lr = 5, 0.1

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step() 
            
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

'''
模型预测
'''
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])


