# Task01：
- 线性回归；
- Softmax与分类模型；
- 多层感知机

## 模型框架：
  模型     数据集    损失函数   优化函数  

## PyTorch语法总结
### 1.安装PyTorch
参考
[本地安装PyTorch](https://blog.csdn.net/qq_39377418/article/details/100336356 "安装Pytorcn_CSDN")
官网
[官网](https://pytorch.org/ " ")
### 2.生成张量
(```）
import torch
n=10
a=torch.ones(n)
b=torch.zeros(n)
c=torch.Tensor(n,n+1) #未初始化的张量
d=torch.rand(n,n+1) #随机初始化的张量
print(d.size()) #查看张量的尺寸
(```)
(```)
#从numpy创建张量
import numpy 
import torch
a = np.array([2.33, 1.07,1.23])
a = torch.from_numpy(a)  # torch.DoubleTensor
print(a)
b = np.array([[3.14],[0.98],[1.32]])
b = torch.from_numpy(b)  # torch.DoubleTensor
print(b)
(```)
