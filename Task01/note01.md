#Task01：
- 线性回归；
- Softmax与分类模型；
- 多层感知机

##模型框架：
  模型     数据集    损失函数   优化函数  

##线性回归：
###  1.生成数据集
使用线性模型来生成数据集，生成一个1000个样本的数据集，下面是用来生成数据的线性关系：

$$
\mathrm{price} = w_{\mathrm{area}} \cdot \mathrm{area} + w_{\mathrm{age}} \cdot \mathrm{age} + b
$$

###  2.读取数据集
