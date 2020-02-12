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
```
#输出Torch版本
print(torch.__version__)
```
### 2.生成张量
```
import torch
n=10
a=torch.ones(n)
b=torch.zeros(n)
c=torch.Tensor(n,n+1) #未初始化的张量
d=torch.rand(n,n+1) #随机初始化的张量
print(d.size()) #查看张量的尺寸
```
```
#从numpy创建张量
import numpy 
import torch
a = np.array([2.33, 1.07,1.23])
a = torch.from_numpy(a)  # torch.DoubleTensor
print(a)
b = np.array([[3.14],[0.98],[1.32]])
b = torch.from_numpy(b)  # torch.DoubleTensor
print(b)
```
```
#tensor转化为array
a=torch.ones(3,2)
a=a.numpy()
```
### 3.tensor加法
[参考](https://www.cnblogs.com/hellcat/p/6850256.html)
```
a = torch.ones(2,2)
b = torch.zeros(2,2)

# 语法一
print(a+b)
# 语法二
print(torch.add(a,b))
# 语法三
print(b.add_(a))
# 语法三
c = torch.Tensor(2,2)
torch.add(a,b,out=c)
print(c)
```
## 1. 线性回归
## 1.1. 从零开始实现
### 1.1.1. 生成数据集
- **torch.randn(*sizes, out=None) → Tensor**[参考](https://blog.csdn.net/infinita_LV/article/details/86546358）
返回一个张量，从标准正态分布（均值为0，方差为1）中抽取的一组随机数。张量的形状由参数sizes定义。

参数:
sizes (int…) - 整数序列，定义了输出张量的形状
out (Tensor, optinal) - 结果张量

```
import torch
torch.randn(2, 3)

0.5419 0.1594 -0.0413
-2.7937 0.9534 0.4561
[torch.FloatTensor of size 2x3]
```
### 1.1.2. 读取数据集
- **random.shuffle()**[参考](https://www.runoob.com/python/func-number-shuffle.html)
```
import random

list = [20, 16, 10, 5]
random.shuffle(list)
print ("随机排序列表 : ",  list)
```
- **torch.LongTensor()**[参考](https://pytorch.apachecn.org/docs/1.0/tensors.html)

这个的Data_Type是64-bit integer (signed)
torch.Tensor 是默认的tensor类型 (torch.FloatTensor) 的简称
一共有八种不同的类型

- **yield**[参考](https://www.jianshu.com/p/d09778f4e055)

for ... in .. 循环中，所有数据都在内存中
带yield的函数是一个生成器generator,用于迭代
类似于return的关键字，迭代一次遇见yield就返回后面的值，重点是：下一次迭代时，从上一次迭代遇到的yield后面的代码(下一行)开始执行。
**简单理解：**yield就是 return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后(下一行)开始。

### 1.1.3. 初始化模型参数

- **numpy.random.normal(loc=0.0, scale=1.0, size=None)**[参考](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.normal.html)
Parameters:	
>loc : 
>>float or array_like of floats
Mean (“centre”) of the distribution.

>scale : 
>>float or array_like of floats
Standard deviation (spread or “width”) of the distribution.

>size : int or tuple of ints, optional
>>Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if loc and scale are both scalars. Otherwise, np.broadcast(loc, scale).size samples are drawn.

Returns:	
>out : 
>>ndarray or scalar
Drawn samples from the parameterized normal distribution




