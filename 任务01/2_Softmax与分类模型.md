#Softmax与分类模型

##基本概念
- 是用于预测种类，预测离散值，用于分类

- 输出数量是标签的类别数，输出值是为该标签的**概率**

- 用softmax运算符（**softmax operator**）--将输出值转化为值为正且和为1的概率分布。

> **单样本矢量计算**
>> 输入----单样本
>> 输出----单样本

> **小批量矢量计算**
>> 输入-----小批量
>> 输出-----小批量

##损失函数
    
##获取训练集和读取数据
- **torchvision包**
它是服务于PyTorch深度学习框架的，主要用来构建计算机视觉模型。torchvision主要由以下几部分构成：
1. torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
2. torchvision.models: 包含常用的**模型结构**（含预训练模型），例如AlexNet、VGG、ResNet等；
3. torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
4. torchvision.utils: 其他的一些有用的方法。

###1.导入包
``` 
import d2lzh_pytorch as d2l #将前面介绍的包封装进去的
```
- 下载d2lzh_pytorch，将它解压在该代码文件夹下
- 下载torchtext
```
pip install torchtext -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- [**import sys**](https://blog.csdn.net/u013203733/article/details/72540075)
sys模块中包含了与python解释器和它的环境有关的函数
当Python执行import sys语句的时候，它在sys.path变量中所列目录中寻找sys.py模块。如果找到了这个文件，这个模块的主块中的语句将被运行，然后这个模块将能够被你 使用 。注意，初始化过程仅在我们 第一次 输入模块的时候进行。另外，“sys”是“system”的缩写。

###2.获取数据集
>**class torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=False)**
>>- root（string）– 数据集的根目录，其中存放processed/training.pt和processed/test.pt文件。
- train（bool, 可选）– 如果设置为True，从training.pt创建数据集，否则从test.pt创建。
- download（bool, 可选）– 如果设置为True，从互联网下载数据并放到root文件夹下。如果root目录下已经存在数据，不会再次下载。
- transform（可被调用 , 可选）– 一种函数或变换，输入PIL图片，返回变换之后的数据。如：transforms.RandomCrop。
- target_transform（可被调用 , 可选）– 一种函数或变换，输入目标，进行变换。

- **四个数据**
http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz

http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

### 3. 从零开始实现

- **import sys** 
#为了引入d2lzh_pytorch包而设置的
import d2lzh_pytorch as d2l

- **对多维Tensor按维度操作**
```
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))  # dim为0，按照相同的列求和，并在结果中保留列特征
print(X.sum(dim=1, keepdim=True))  # dim为1，按照相同的行求和，并在结果中保留行特征
print(X.sum(dim=0, keepdim=False)) # dim为0，按照相同的列求和，不在结果中保留列特征
print(X.sum(dim=1, keepdim=False)) # dim为1，按照相同的行求和，不在结果中保留行特征
```
- **广播机制**
要么两个的shape相同，要么其中一个为1

- **y_hat.gather**(1, y.view(-1, 1))----（=1按行取，如何去数据分别代表第一行的第几个-第二行的第几个）
取数据，按照后面的从前面y_hat数据中取

y.**view**(-1, 1)   将行向量变为列向量了，-1就是自己算，1就是1列

- y_hat.**argmax**(dim=1)
取最大的数，dim=1，按照行取每行最大


### 4. 简洁实现
- num_inputs = 784#28*28 784个 输入特征

- class FlattenLayer(nn.Module):#输入特征转换 将28*28转化为784

-  OrderedDict([
           ('flatten', FlattenLayer()),
           ('linear', nn.Linear(num_inputs, num_outputs))]) # 或者写成我们自己定义的 LinearNet(num_inputs, num_outputs) 也可以
        )

一层变换层，一层线性层

- loss = **nn.CrossEntropyLoss() **
交叉熵损失函数

- **torch.optim.SGD**
随机梯度下降


