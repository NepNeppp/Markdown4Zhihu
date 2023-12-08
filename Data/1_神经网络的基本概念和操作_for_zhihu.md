# ==Windows 下的环境搭建==

1.确认有Nvidia的GPU

可以通过win+R `dxdiag`查看显卡信息

![image-20211129004602255](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211129004602255.png)

## 1.下载并安装CUDA

https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local

（关闭360）

安装完成后命令行输入 `nvidia-smi`

![image-20211128133807083](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211128133807083.png)

## 2.配置python环境

### 安装miniconda

由于已经安装了anaconda 这里就不跟沐神安装miniconda了



### 安装GPU版本pytorch

打开anaconda powershell prompt  输入`python`查看版本（ 3.8.8）

进入pytorch官网 https://pytorch.org/get-started/locally/

![image-20211128134418568](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211128134418568.png)

（由于pytorch还没有11.5的版本 无所谓啦）



复制后在anaconda powershell prompt中输入生成的命令（如果之前查看了python版本记得退出python）

`pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

![image-20211128144137815](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211128144137815.png)



太慢了最终失败：使用镜像

使用windows的**批处理程序**，编写文件anaconda_add_package_sources.bat

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2


conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/simpleitk

conda config --set show_channel_urls yes
```

在anaconda powershell prompt中输入 type C:\Users\Lenovo\Desktop\anaconda_add_package_sources.bat | cmd

输入conda config --show-sources命令查看包源

![image-20211128153554008](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211128153554008.png)

接下来在进入pytorch官网找到安装命令，去掉-c之后的内容（`-c`是指定从哪个源来下载我们要的包，但是我们用的是国内源，我们上一步已经配置，这记录在系统中了，而且在源使用列表中的最前面，所以conda能够自动使用国内源）

这里我们安装11.1，找previous：![image-20211128175828331](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211128175828331.png)

`conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge`

即输入 `conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1`

等一会输入 y

![image-20211128154603078](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211128154603078.png)

睡一觉后~

![image-20211128173718455](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211128173718455.png)

验证是否成功：在anaconda powershell prompt中输入`python`，然后输入

```python
import torch
a = torch.ones((3,1))      #创建全为1的a矩阵
a = a.cuda(0)			   #把a移动到cuda0的位置，即GPU上
b = torch.ones.((3,1)).cuda(0)
a + b
```

![image-20211128175112170](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211128175112170.png)

device ='cuda:0' 成功！



### 安装Jupyer和d2l

由于之前已经安装了jupyter notebook 现在只需安装d2l `PS C:\Users\Lenovo> pip install d2l`

可以通过`pip -show d2l` 查看d2l包的位置



## 3.尝试运行案例

最后运行代码试试

notebook找到 d2l-zh/pytorch/chapter_convolutional-modern/resnet.ipynb   选择kernel→Run All

调整一下**batch_size** ，不然GPU内存不够

![image-20211130145728268](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211130145728268.png)

GPU起飞！

![image-20211130145501095](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211130145501095.png)



没有对比就没有伤害（左李沐游戏本，右GF游戏本）

沐神：

![image-20211130151229806](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211130151229806.png)

Kang：

![image-20211130151121407](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211130151121407.png)

大功告成！==Oss！==

完事后记得**清除显卡内存**，通过`nvidia-smi` （也就是我的Lenovo下nep-smi→nep）查看对应python的进程 pid   再输入`tskill pid`





# ==线性回归==

## 小批量随机梯度下降

在整个训练集上计算梯度太贵了（恩达：i从1到m） 一个深度神经网络模型可能需要数分钟甚至数小时
故我们可以随机采样b个样本来近似损失

<img src="https://www.zhihu.com/equation?tex=for \quad b \quad examples: i_1,i_2,...,i_b \\
\frac 1 b \sum_{i \in I_b} l({\bf x}_i, y_i, {\bf w})
" alt="for \quad b \quad examples: i_1,i_2,...,i_b \\
\frac 1 b \sum_{i \in I_b} l({\bf x}_i, y_i, {\bf w})
" class="ee_img tr_noresize" eeimg="1">
选择批量大小：超参数b不能太大（内存）也不能太小（计算量小，不适合并行来最大利用计算资源）



## 从零实现线性回归

(**小批量随机梯度下降**)

包括数据流水线、模型、损失函数和小批量随机梯度下降优化器（不使用框架）
便于从底层了解

步骤：

1. 构造人工数据集`synthetic_data`: 根据接受的`w` 和 `b` 以及样本个数`num_examples`， 生成数据集，并添加噪声

2. 数据处理`data_iter`：

   根据接受的 批量大小`batch_size`，特征矩阵`X`，标签向量`y` ，生成大小为`batch_size`的小批量

3. 定义模型

   + 定义模型
   + 定义损失函数
   + 定义优化算法

4. 训练模型



**导包：**

```python
%matplotlib inline  # 魔法函数（Magic Function）内嵌绘图，并且可以省略掉plt.show()
import random     #随机梯度下降&初始化权重
import torch
from d2l import torch as d2l  #Mu已经弄好的包
```



**构造人造数据集：**根据带有噪声的线性模型构造一个人工数据集。我们使用线性模型参数 **w**=[2,-3.4]^T^, b = 4.2和噪声项ε生成数据集及其标签

<img src="https://www.zhihu.com/equation?tex={\bf y} = {\bf Xw} + b + \epsilon
" alt="{\bf y} = {\bf Xw} + b + \epsilon
" class="ee_img tr_noresize" eeimg="1">

```python
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + noise"""
    X = torch.normal(0, 1, (num_examples, len(w)))  #标准正态 ，num_examples个样本，len(w)列
    y = torch.matmul(X,w) + b
    y += torch.normal(0, 0.01, y.shape)  #noise
    return X, y.reshape((-1,1))  #将y变成列向量  -1即automatic n
    
true_w = torch.tensor([2, -3.4])
true_b = 4.2

features, labels = synthetic_data(true_w, true_b, 1000)
```



**batching：**将整个函数随机分为很多个batches，并返回每一个batch

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))  #生成每个样本的index，转为py的list
    # 使这些样本是随机读取的，没有特定顺序 random.shuffle(indices)使对应下标indices随机排序
    random.shuffle(indices)
    
    #根据batch_size切成很多batches
    for i in range(0, num_examples, batch_size):   #这都忘了？！0到examps，间隔为batch_size
        #一batch一batch地取出来
        batch_indices = torch.tensor(
                        indices[i:min(i + batch_size, num_examples)])  # 考虑最后一块不一定是有batch_size个样本
        yield features[batch_indices], labels[batch_indices] # 每一次都返回一个batch,外面用for...in循环接
        
batch_size = 10
    
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```



**定义模型**

```python
#初始化参数
w = torch.normal(0,0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#定义模型
def linreg(X, w, b):
    '''线性回归模型。'''
    return torch.matmul(X, w) + b

#定义损失函数(Vectorization)
def squared_loss(y_hat, y):
    '''均方损失'''
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

#定义优化算法
def sgd(params, lr, batch_size):
    '''小批量随机梯度下降'''
    with torch.no_grad():   #不计算梯度，即更新时不参与梯度计算（保留之前计算的梯度）
        for param in params:
            param -= lr * param.grad / batch_size   #求均值（之前的loss没有 /10）
            param.grad.zero_()   #梯度置为0，不影响下一轮计算
```



**训练模型**

```python
lr = 0.03
num_epochs = 3  #整个数据扫三遍
net = linreg    #指定net和loss 便于之后更换模型 我靠py方便啊！
loss = squared_loss

# 其他模型训练实现大同小异
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)    # X 和 y的小批量损失, 待会要的也是参数这一步的梯度
        #因为 l 的形状是(batch_size,1),而不是一个标量
        #故让 l 中所有的元素加到一起 来计算[w,b]的梯度(一个长长的向量，盲猜神经网络参数在梯度下降更新时也是用一个向量表示，统一接口
        l.sum().backward()
        sgd([w,b], lr, batch_size) #使用参数的梯度更新参数 （由于都是除以batch_size，最后一个batch很可能会被多除）
    #输出每一次扫描数据后的loss，故此计算不需要更新梯度
    with torch.no_grad():
        train_l = loss(net(X, w, b), y)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```



**与真实值进行比较**

```python
print(f'w的误差： {true_w - w.reshape(true_w.shape)}')
print(f'b的误差： {true_b -b}')
```



调整超参数 lr 和 batch_size，直观感受其影响（调整后记得要重新初始化w，这样就不会继承之前的梯度结果）



## 线性回归的简单实现

**君子性非异也，善假于物也**



导包，数据初始化

```python
import numpy as np
import torch
from torch.utils import data   # 处理数据的模块
from d2l import torch as d2l

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000) #跟我自己写的synthetic_data()应该差不多
```



通过调用框架中现有的API来读取数据
一样的根据batch_size一次一次地给我数据，返回等价于从零开始的 data_iter

```python
def laod_array(data_arrays, batch_size, is_train=True):
    '''构造一个pytorch数据迭代器'''
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)   #shuffle 表示是否需要打乱顺序

batch_size = 10
#把features和labels做成一个list传到data.TensorDataset()里面得到dataset
data_iter = laod_array((features, labels), batch_size)   

next(iter(data_iter))
```



使用框架的预定义好的层

```python
from torch import nn

net = nn.Sequential(nn.Linear(2,1))   # 线性层（全连接层），2个神经元的模型，放到Sequential容器中（list of layers）
```



初始化模型参数

```python
net[0].weight.data.normal_(0,0.01)  #使用正态分布来替换掉 0层 权重w  真实data 的值
net[0].bias.data.fill_(0)
```



计算均方误差使用MSELoss类，也称平方L2范数

```python
loss = nn.MSELoss()
```



实例化SGD

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```



训练代码跟之前的从零开始非常相似

```python
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X),y)
        trainer.zero_grad()   #让优化器先把梯度清零
        l.backward()      #pytorch已经帮我求sum了
        trainer.step()    #使用step函数进行一次模型的更新
        
    l = loss(net(features),labels)  #没有再清零了，反正也没run backward()
    print(f'epoch {epoch + 1}, loss {l:f}')
```



## QA环节



Q：batch_size是否会影响最终模型结果？

A：实际上过小是好的，同样扫数据10次的情况下，batch_size越小对结果的收敛越好。（噪音对神经网络反而是一件好事情，一定的噪音使得神经网络不容易走偏，对小孩严厉点！）

宏毅：加入GPU并行计算后，batch_size大的反而会计算的更快一点（计算偏导速度差不多，但batch_size小会更新参数更慢），batch_size小可以使得梯度下降的方向不单一（防止限死在不好的local minimum，即峡谷），batch_size小更容易收敛到平原（能更好的match测试集，模型偏差后，损失不会偏差太大）



Q：随机梯度下降的“随机”是指什么？

A：是指随机采样batch_size个元素



Q：为什么机器学习优化算法都采用梯度下降（一阶导算法），而不采用牛顿法（二阶导算法），（收敛更快）

A：如果一阶导是100维度的向量，二阶导就是100×100的矩阵(贵)，比起收敛速度，我们更在乎模型的正确性，即收敛到哪个地方。

人生亦如此，步子不要迈太大急于求成，关键是想走到哪里



Q：这样的data_iter写法，每次都把所有输入load进去，数据多了内存会不会爆掉？

A：数据特别大会爆掉，整本书的dataset相对比较小，GPU机器内存几个G也还是够。

实际情况数据在硬盘上，每一次是在硬盘上随机抓出来，为了性能还可以预取



Q：使用生成器生成数据有什么优势？相比return

A：不用一次把所有batch取出来要一个batch取一个batch；python习惯



Q：如果样本大小不是batch_size的整数倍？

A：3种方法  1.拿一个小点的batch  2.丢掉    3.在其他地方拿一点补全



Q：收敛的判断方法？

A：两个函数变化不大时（达到阈值）停 或 验证数据集达到要求 停



Q：为什么会出现NAN？

A：可能是因为求导时出现了0 (之后会讲数值稳定性)



# ==Softmax回归==

实际上是多元分类问题，输出i为第i类的置信度

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211204182201827.png" alt="image-20211204182201827" style="zoom:50%;" />

+ 类别进行一位有效编码，**one-hot**编码的思想：


<img src="https://www.zhihu.com/equation?tex={\bf y} = [y_1,y_2,...,y_n]^T\\
y_i = 
\begin{cases}
1 ,&if \quad i = y\\0,&otherwise
\end{cases}
" alt="{\bf y} = [y_1,y_2,...,y_n]^T\\
y_i = 
\begin{cases}
1 ,&if \quad i = y\\0,&otherwise
\end{cases}
" class="ee_img tr_noresize" eeimg="1">

+ 使用均方损失训练
+ 输出最大值为预测


<img src="https://www.zhihu.com/equation?tex=\hat{y} = argmax\, \{o_i\}
" alt="\hat{y} = argmax\, \{o_i\}
" class="ee_img tr_noresize" eeimg="1">

+ 需要更置信的识别正确类（大余量），即预测值与其他非预测指差距要明显。
+ 输出的置信度匹配概率（非负，和为1，王木头yyds！）


<img src="https://www.zhihu.com/equation?tex={\bf \hat{y}} = softmax({\bf o})\\
\hat{y}_i = \frac {\exp(o_i)} {\sum_k \exp(o_k)}
" alt="{\bf \hat{y}} = softmax({\bf o})\\
\hat{y}_i = \frac {\exp(o_i)} {\sum_k \exp(o_k)}
" class="ee_img tr_noresize" eeimg="1">

+ 交叉熵（Cross Entropy）通常用来衡量两个概率（预测和真实）的区别


<img src="https://www.zhihu.com/equation?tex=l({\bf y,\hat{y}}) = -\sum_i y_i\log\hat{y}_i = -\log\hat{y}_y
" alt="l({\bf y,\hat{y}}) = -\sum_i y_i\log\hat{y}_i = -\log\hat{y}_y
" class="ee_img tr_noresize" eeimg="1">

注：这里我们只考虑它对正确的那一类的预测为正确的情况（吴恩达还考虑了将错误的类预测为错误的情况）

Mu：对于分类问题来讲，我们不关心对非正确类的预测值，我们只关心对正确类的预测值的支持度有多大

+ 其梯度是真实概率与预测概率的区别 ？？？？？


<img src="https://www.zhihu.com/equation?tex=\partial_{o_i}l({\bf y,\hat{y}}) = softmax({\bf o})_i - y_i
" alt="\partial_{o_i}l({\bf y,\hat{y}}) = softmax({\bf o})_i - y_i
" class="ee_img tr_noresize" eeimg="1">



注：sigmoid就是将0和输出z进行softmax的过程



## *损失函数



### L~2~ Loss

即均方损失

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211205003458407.png" alt="image-20211205003458407" style="zoom: 50%;" />

如图，蓝线表示y为0时，l随着预测值的变化

绿线代表其最大似然估计，其值为 e^-l^ ，它的似然函数就是一个高斯分布

橙线代表损失函数的梯度



梯度下降的时候我们从负梯度方向来更新参数，由图可知，**当预测值y’ 与真实值y 相差比较大的时候，那么梯度绝对值比较大（即对参数更新的幅度很大**，这多不一定是一件美事儿啊！）；y‘ 与y 靠近的时候，梯度绝对值会越来越小

故提出改进的 L~1~ Loss



### L~1~ Loss

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211205005104760.png" alt="image-20211205005104760" style="zoom:50%;" />

蓝线表示其y=0时的损失函数

绿线为似然函数

橙线为导数（绝对值函数在零点不可导，但其sub gradient可以在-1到+1之间）



即使预测值y’ 与真实值y 相差比较大的时候，更新的权重也不会特别大（稳定性）

但当预测值y’ 与真实值y 相差很小的时候，可能会震荡，不够稳定



### Huber’s Robust Loss

结合L~1~与 L~2~各自的优点，我们希望当预测值和真实值相差比较大的时候，梯度不要太大。相差比较小的时候，梯度也要比较小。

<img src="https://www.zhihu.com/equation?tex=l(y,y')=
\begin{cases}
|y-y'| - \frac 1 2 \quad &if |y-y'|>1 \\
\frac 1 2 (y-y')^2 &otherwise
\end{cases}
" alt="l(y,y')=
\begin{cases}
|y-y'| - \frac 1 2 \quad &if |y-y'|>1 \\
\frac 1 2 (y-y')^2 &otherwise
\end{cases}
" class="ee_img tr_noresize" eeimg="1">
<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211205011047696.png" alt="image-20211205011047696" style="zoom:50%;" />

第一项 -1/2 是为了让两条曲线连接起来

这样当预测值与真实值相差较大时，梯度也不会太大；而二者相差比较小的时候，梯度也比较小



## *读取图像分类数据集



NMIST数据集是图像分类中广泛实用的数据集之一，但作为基准数据集过于简单(198几年提出来的，非常经典，cv界的hello world)，我们将使用类似但更复杂的**Fashion-MNIST数据集**

```python
%matplotlib inline
import torch
import torchvision
from torch.utils import data   #又来了，处理数据的模块，方便读取数据
from torchvision import transforms # 对数据操作的
from d2l import torch as d2l

d2l.use_svg_display()  #用svg来显示图片
```



通过框架中的内置函数将Fashion-mnist数据集下载并读取到内存中

```python
# 通过ToTensor实例将图像数据从PIL类型变换为32位浮点数格式
# 并除以255使得所有像素的均值在0-1之间（feature scaling）
trans = transforms.ToTensor()

# root是下载文件的位置 （跟linux一样 .当前目录 ..父目录 ./当前目录的文件或文件夹 ../父目录的文件或文件夹）
# trian表示是训练集还是测试集，transform=trans为了拿到的是tensor而不是图片，download=True表示默认从网上下载
mnist_train = torchvision.datasets.FashionMNIST(root = '.\data', train=True,
                                               transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root = '.data', train = False,
                                               transform=trans, download=True)

len(mnist_train), len(mnist_test)
```

![image-20211205015905945](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211205015905945.png)



定义两个可视化数据集的函数（哈啦休！）

```python
# 将Fashion-MNIST中的10个类别的数字索引与其对应的文本名称进行转换
def get_fashion_mnist_labels(labels):
    '''返回Fashion_MNIST数据集的文本标签'''
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal',
                   'shirt', 'senaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 使用matplotlib来画几张图
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    '''plot a list of images.'''
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

几个样本的图像及其相应的标签

```python
X, y = next(iter(data.DataLoader(mnist_train, batch_size = 18)))
show_images(X.reshape(18,28,28), 2, 9, titles=get_fashion_mnist_labels(y))
```

![image-20211205211244376](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211205211244376.png)



多进程读取以小批量数据，大小为`batch_size`
读数据的速度至少要比训练的速度快，快多一点更好

```py
batch_size = 256

def get_dataloader_workers():
    '''使用4个进程来读取数据'''
    return 4

# 据说要用`if __name__ == "__main__":`才可以哦，这样可以使得只有运行该文件时才会run以下内容，而以模块形式被import时不会被调用
# if __name__ == "__main__":
# 训练集数据  batch_size 是否打乱顺序（训练集需要，测试集就不一定了） 多进程
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=get_dataloader_workers())

# 用Timer来测试速度
timer = d2l.Timer()

# 一个一个访问所有的batch

for X, y in train_iter:
    continue
        
f'{timer.stop():.2f} sec'
```

？？？？？有时候报错有时候不报错，对多进程等的理解8够啊



### 整合所有组件
定义`load_data_fashion_mnist`函数，用于获取和读取数据集。它返回训练集和验证集的数据迭代器。此外还和接受一个可选参数来对图片大小进行调整

为什么`trans = [transforms.ToTensor()]` 要加个`[]`，之前都没加 ？？？？？

```python
# 更换数据集后可以通过resize改变图片大小
def load_data_fashion_mnist(batch_size, resize=None):
    '''下载Fashion-MNIST数据集，然后将其加入到内存'''
    # 为什么这里要加个`[ ]`？？？？？
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='./data',
                                                   train=True,
                                                   transform=trans,
                                                   download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='./data',
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
           data.DataLoader(mnist_test, batch_size, shuffle=False,
                          num_workers=get_dataloader_workers()))
```

通过指定resize参数来测试`load_data_fashion_mnist`函数的图像大小调整功能

```python
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

```
Out[nep]: torch.Size([32, 1, 64, 64]) torch.float32 torch.Size([32]) torch.int64
```



## softmax回归从零开始的异世界生活

**导包，初始化参数**

```python
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

# 初始化参数
# 要对长度784的向量进行10种线性运算（对应10个类别，故系数矩阵为784×10）
# 每一个类别的线性运算对应一个bias
W = torch.normal(0,0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```



**实现softmax运算**

```python
# X是一个矩阵，每行代表一个样本（也就是batch_size），每列代表其分类（10个分类）。
# 对一个矩阵用softmax（之前只想到一维向量，拓展到n个样本）
# 按列求和，合并成一个列向量，将该列向量作为分母，利用广播机制（把列向量复制脑补成X的shape再按元素操作）
# 即对X每行进行一次softmax
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition
```

测试一下
我们将每个元素变成一个非负数，此外依据概率原理，每行总和要为1

```python
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(1)
```

```
Out[nep]: 
(tensor([[0.2933, 0.0329, 0.6146, 0.0508, 0.0083],
         [0.4262, 0.1930, 0.0895, 0.2288, 0.0626]]),
 tensor([1., 1.]))
```

**实现softmax回归模型**

```python
def net(X):
    # 把输入reshape为 n×784 （这里n=batch_size也就是256），与系数矩阵W（784×10）相乘
    # 再加bias b（1×10，广播机制）
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)
```



实现交叉熵损失函数

怎么拿出预测的概率呢？

**通过索引取值**

例： 创建一个数据`y_hat`,其中包含2个样本在3个类别的预测概率，使用`y`作为`y_hat`中概率的索引（`y`即为所有样本真实值对应的类型组成的向量）

```python
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3 ,0.6], [0.3, 0.2, 0.5]])
# 按索引取值, 前一个向量代表从几号样本拿（行），后面的向量代表那对应样本的第几个元素（列）
y_hat[[0, 1], [0, 2]]

# Out[nep]: tensor([0.1000, 0.5000])
```

**交叉熵损失函数**

```python
def cross_entropy(y_hat, y):
    # 通过索引，拿出y_hat中每个样本对其的真实类别的预测概率
    return -1 * torch.log(y_hat[range(len(y_hat)), y])
cross_entropy(y_hat, y)
```



**将`y_hat`（矩阵）与`y`（向量，其值代表标签，也是y_hat中真实类别的索引 ↑ ）进行比较，计算预测正确数目**

```python
def accuracy(y_hat, y):
    '''计算预测正确的数量'''
    # 对于二维矩阵y_hat，行大于1列大于1
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 选每一行的最大值的下标
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    # print(cmp.type(y.dtype))
    return float(cmp.type(y.dtype).sum())
```

```python
# kang: 利用之前讲的索引和argmax，可选取矩阵每一行的最大值
y_hat[range(len(y_hat)), y_hat.argmax(axis=1)]
```

**对于任意数据迭代器可以访问的数据集，我们可以评估任意模型`net`的准确率**（一个迭代周期）

```python
def evaluate_accuracy(net, data_iter):
    '''计算在指定数据集上模型的准确率'''
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
```

**实现累加器**，`Accumulator`实例中创建了两个变量，用于分别存储正确预测数和预测总数
\_\_init\_\_() ？？？？？
\_\_getitem\_\_()

```python
class Accumulator:
    '''在n个变量上累加'''
    def __init__(self, n):
        self.data = [0.0] * n
        
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        
    def reset(self):
        self.data = [0.0] * len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
```



**Softmax回归的训练**
updater是更新模型参数的常⽤函数，它接受批量⼤小作为参数。它可以是封装的d2l.sgd函数，可以是框架的内置优化函数

**一个迭代周期的训练**
返回一个迭代周期训练的损失和训练的精度

```python
def train_epoch_ch3(net, train_iter, loss, updater):
    '''训练模型的一个迭代周期'''
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
        
    # 训练损失总和、训练准确数总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用pytorchn内置的优化器和损失函数
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y),
                      y.size().numel())
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        # 返回训练损失和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]
```

定义一个在动画中绘制数据的实用程序类

```python
class Animator:
    '''在动画中绘制数据'''
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                ylim=None, xscale='linear', yscale='linear',
                fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        #  使用lambda函数捕捉参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    
    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, '__len__'):
            y = [y]
        n = len(y)
        if not hasattr(x, '__len__'):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x,y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

**训练函数**，它会在train_iter访问到的数据集上训练一个模型net。该训练函数会运行多个迭代周期（由mun_epochs指定）。在每个迭代周期结束时，利用text_iter访问到的测试数据集对模型进行评估。最后利用刚刚的Animator类来可视化训练进度

```python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    '''第三章的训练模型，之后还会复杂'''
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                       legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <=1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

小批量随机梯度下降来优化模型的损失函数（也就是自己写updater）

```python
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

训练10个迭代周期

```python
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

第二次迭代后结果

![](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211208010422871.png)



**预测**

```python
def predict_ch3(net, test_iter, n=6):
    '''预测标签'''
    i = 0
    for X, y in test_iter:
        i += 1
        if i < 11:
            trues = d2l.get_fashion_mnist_labels(y)
            preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
            titles = [ true + '\n It\'s ' + pred for true, pred in zip(trues, preds)]
            d2l.show_images(X[0:n].reshape((n,28,28)), 1, n, titles=titles[0:n])
        else:
            break
predict_ch3(net, test_iter)
```

正确case：<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211208010558855.png" alt="image-20211208010558855" style="zoom: 80%;" />

错误case：![image-20211208010853538](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211208010853538.png)





## softmax的简洁实现

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```



softmax的输出是一个全连接，因此我们只需要在Sequential中添加一个带有10个输出的全连接层

```python
# pytorch不会隐式地调整输入的形状，因此，
# 我们在线性层前定义展平层（flatten）来调整网络输入的形状(把输入784输出10的线性模型展平,对应之前的reshape)
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

# 对每一个layer m 做一次
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```



之前是先进行softmax再计算交叉熵，但由于指数可能导致数据溢出（overflow＆underflow），故，使用归一化后的数据计算交叉熵
即向交叉熵损失函数中传递未归一化的预测，并同时计算softmax及其对数（交叉熵），这是一件聪明的事情“LogSumExp技巧”

所以说pytorch在做cross entropy loss的计算时会自动帮你softmax

```python
loss = nn.CrossEntropyLoss()
```

使用学习率为0.1的小批量随机梯度下降作为优化算法对参数进行调整

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

调用之前定义的训练函数来训练模型

```python
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```



![image-20211209120949641](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211209120949641.png)



由于之前是在loss里面用的softmax，故模型本身并没有进行归一化处理，验证一下：

![image-20211214011421031](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211214011421031.png)



## QA环节

Q：**softlabel**训练策略？

A：softmax由于使用指数，其概率很难逼近一个0或者1的数（z要趋于无穷）。故我们可以将正确类记为0.9，其余不正确的类平分0.1，这样可以使得softmax能够很好地拟合0.9和那些很小的数。



Q：为什么使用交叉熵而不用相对熵、互信息等其他基于信息量的度量？

A：交叉熵是对称的，相对熵（KL散度）不对称，互信息难算。（理论上都可以，主要看好不好算，就像昨晚21.12.8想的为什么不用矩估计来确定参数一样）



Q：为什么交叉熵 ylogy^ 我们只关心了正确的类，没有关心不正确的类（和吴恩达的不一样）？ 

A：因为之前的y对不正确的类的概率为0，故不关心。如果使用softlabel，就需要关心了（nepnep？好像还是跟吴恩达的不一样）



Q：对于n分类问题，只有1个正类，n-1个负类，会不会类别不平衡？

A：如果使用0,1编码的话可以不用关心负类，只关心正类（不用关心是不是不平衡，而是关心**每一个类都要有足够多的样本**）



Q：之前讲解Loss时的似然函数是怎么得出来的？有什么参考价值？

A：**机器学习后面的模型跟统计学关系不是特别大**。最小loss等价于最大似然



Q：`Dataloader()`的`num_workers`是并行吗？进程数可以直接写吗，为什么要专门写个函数？

A：是的，pytorch会在后端开num_workers个进程进行并行。早一点版本Windows不支持多进程。



Q：为什么不在`accuracy(y_hat, y)`函数中把除以`len(y)`做完，而是要返回数量再定义`evaluate_accuracy(net, data_iter)`函数来计算准确率呢？

A：防止最后一个batch不够batch_size，导致最后一个batch被多除



Q：在计算精度的时候，为什么要使用`net,eval()`将模型设置为评估模式？

A：不做梯度计算，提高性能



Q：**自己的图片数据集**，应该怎么处理才能用于训练？怎么根据自己图片训练集和测试集创建迭代器？

A：具体可以通过查看pytorch文档 （把图片按分类放在对应文件夹下，pytorch会根据上一层目录对其扫描读入



# ==多层感知机==

## 感知机

二分类模型，最早的AI模型之一

它的求解算法等价于batch_size为1的梯度下降

它不能拟合XOR函数，导致第一次AI寒冬



## 多层感知机

### 激活函数

激活函数使得感知机脱胎换骨（线性→非线性）

如果没有激活函数，那么无论有多少层，也相当于一个感知机（最终还是线性模型）



**sigmoid函数**

将定义域在R上的输入变换为(0, 1)上的输出，故也叫挤压函数（squashing function）：它将(-inf, inf)中任意输入压缩到区间(0, 1)中的某个值

<img src="https://www.zhihu.com/equation?tex=sigmoid(z) = \frac 1 {1+e^{-z}}
" alt="sigmoid(z) = \frac 1 {1+e^{-z}}
" class="ee_img tr_noresize" eeimg="1">
sigmoid函数的导数为

<img src="https://www.zhihu.com/equation?tex=\frac d {dz}sigmoid(z) = \frac {-e^{-z}} {(1-e^{-z})^2}= \frac {1-e^{-z}+1} {(1-e^{-z})^2}=sigmoid(z)(1-sigmoid(z))
" alt="\frac d {dz}sigmoid(z) = \frac {-e^{-z}} {(1-e^{-z})^2}= \frac {1-e^{-z}+1} {(1-e^{-z})^2}=sigmoid(z)(1-sigmoid(z))
" class="ee_img tr_noresize" eeimg="1">


绘制sigmoid函数图像

```python
%matplotlib inline
import torch
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212175125393.png" alt="image-20211212175125393" style="zoom:80%;" />

求导函数

```python
# 清除以前的梯度（养成习惯，不然会累加）
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid(x)', figsize=(5, 2.5))
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212175157650.png" alt="image-20211212175157650" style="zoom:80%;" />



**tanh函数**

与sigmoid类似，tanh（双曲正切函数）将输入压缩转换到区间(-1, 1)上

<img src="https://www.zhihu.com/equation?tex=\tanh(z)=\frac {1-e^{-2z}} {1+e^{-2z}}
" alt="\tanh(z)=\frac {1-e^{-2z}} {1+e^{-2z}}
" class="ee_img tr_noresize" eeimg="1">
其导数为

<img src="https://www.zhihu.com/equation?tex=\frac d {dz} \tanh(z) = 1-\tanh^2(z)
" alt="\frac d {dz} \tanh(z) = 1-\tanh^2(z)
" class="ee_img tr_noresize" eeimg="1">


绘制tanh函数图像

```python
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.tanh(x)

d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212181155438.png" alt="image-20211212181155438" style="zoom:80%;" />

求导函数

```python
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh(x)', figsize=(5, 2.5))
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212181139868.png" alt="image-20211212181139868" style="zoom:80%;" />

**ReLU函数**

最受欢迎的选择是线性整流单元（Rectified linear unit，ReLU），简单、表现良好，关键是好算！！！

ReLU提供了一种非常简单的线性变换，给定元素z，ReLU函数为

<img src="https://www.zhihu.com/equation?tex={\text ReLU}(x) = \max{(0,x)} 
" alt="{\text ReLU}(x) = \max{(0,x)} 
" class="ee_img tr_noresize" eeimg="1">
==不搞花里胡哨==，相对sigmoid和tanh，不进行指数运算（对cpu来说一次指数大概相当于100次乘法），而且也达到了非线性化的效果



同样绘制其函数和导函数图

```
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)

d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212181809300.png" alt="image-20211212181809300" style="zoom:80%;" />

```python
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu(x)', figsize=(5, 2.5))
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212181832095.png" alt="image-20211212181832095" style="zoom: 80%;" />



### 多隐藏层

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212183152291.png" alt="image-20211212183152291" style="zoom: 50%;" />

设计思路：

+ 少层数多个数 ：例如对于128的输入，可以就一层hidden layer，m=128或256
+ 多层数，减少每层个数：例如对于128的输入，可以3层hidden layer，m1=128， m2=64， m3=32，具体根据模型复杂度判断，一般来说每层递减（机器学习本质上就是对复杂的数据进行压缩，之后的CNN可能会先压缩后扩张）



## 多层感知机的从零开始的实现

导包，导入数据集（通过迭代器每次返回batch_size大小的数据）

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```



回想⼀下，Fashion-MNIST中的每个图像由28 × 28 = 784个灰度像素值组成。所有图像共为10个类别（这是kang改变不了的）。忽略像素之间的空间结构，我们可以将每个图像视为具有784个输⼊特征和10个类的简单分类据集。⾸先，我们将实现⼀个具有`单隐藏层`的多层感知机，它包含`256个隐藏单元`。注意，我们可以将这两个量都视为超参数。**通常，我们选择2的若⼲次幂作为层的宽度。因为内存在硬件中的分配和址⽅式，这么做往往可以在计算上更⾼效**

*可以理解为通过第一层把784维度压缩成了256维度，第二层把256维度压缩成了10维度*
*==彻底搞懂训练时的数据矩阵，系数矩阵==*
*训练时，数据的矩阵 batch_size×num_inputs (乘W1加b1)→ batch_size×num_hiddends (乘W2加b3)→ batch_size×num_outputs*
*那对应的W1不就是 num_inputs×num_hiddend吗，W2不就是num_hiddend×num_outputs吗*
*b1不就是一个num_hiddends的行向量么(对batch_size维度使用广播机制)，同样的b2不就是一个num_outputs的行向量么！*
*这有啥值得纠结的？到时候就是一个转置不转置的问题，重点不在这*

```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 手动设置参数矩阵，累死
# 784行，256列！！！（跟吴恩达的参数矩阵格式一样啊！别忘记了）
# 这里参数矩阵的每一列对应对输入做的一种线性组合
# 如果不用randn，全设为1会怎么样？吴恩达讲的随机初始化，梯度下降时，每个感知机参数的改变会完全相同，相当于该层所有感知机都一样
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)


# kang想用正态分布初始化W
# W1 = nn.Parameter(torch.normal(0, 0.1, (num_inputs, num_hiddens),
#                     dtype=torch.float32, requires_grad=True))
# W2 = nn.Parameter(torch.normal(0, 0.1, (num_hiddens, num_outputs),
#                     dtype=torch.float32, requires_grad=True))

#将W全部设为1
# W1 = nn.Parameter(torch.zeros((num_inputs, num_hiddens),requires_grad= True))
# W2 = nn.Parameter(torch.zeros((num_hiddens, num_outputs),requires_grad= True))

# b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```



实现ReLU激活函数

```python
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```



实现模型

```python
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)   #压缩到了256个维度
    return(H @ W2 + b2)    #压缩到了101个维度
```



损失函数

```python
loss = nn.CrossEntropyLoss()
```



训练

```python
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212200824611.png" alt="image-20211212200824611" style="zoom: 80%;" />

相对于之前的直接softmax，损失更低一点，不过精度差不太多

可见MLP即使进行较大的结构改变，模型差距不一定会很大，故MLP效果不好可以转卷积转RNN转transfoemer但不会改变太多东西（相对于SVN的优点，相对来说需要改变很多东西）



### Kang动手脑洞一下

#### 正态分布初始化参数

把之前的参数初始化换为正态分布会怎么样？（估计没什么变化）



<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212201952369.png" alt="image-20211212201952369" style="zoom: 80%;" />

loss太大了！！！ 是不是因为正态分布的方差太大了呢？~~（相对来说learning rate就太小了）~~

**修改参数初始化的正态分布方差**为0.1：<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212202345913.png" alt="image-20211212202345913" style="zoom: 80%;" />

此时效果和最初差不多

或者不改变方差，将**learn rate**设为1：

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212202614592.png" alt="image-20211212202614592" style="zoom: 80%;" />

lr设为3：

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212202735779.png" alt="image-20211212202735779" style="zoom: 80%;" />

果然是大了，设为0.3：

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212202938523.png" alt="image-20211212202938523" style="zoom: 80%;" />

~~Yes！可以减小初始参数的大小或者改变learn rate~~

分析原因：由于一开始随机初始化时正态分布的方差过大，导致对不同参数的调整到位所需要的**epochs**不一样，现在猜测对于正态分布初始化方差为1的系数，使用lr=0.3时，epochs要延长到20/50

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212204338133.png" alt="image-20211212204338133" style="zoom: 80%;" />

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212214439837.png" alt="image-20211212214439837" style="zoom: 80%;" />

==感悟：不能只局限在修改一个超参数上，有可能需要同步调整==

作为对比，randn初始化很小的参数W，lr=0.1，epochs=50

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212215608886.png" alt="image-20211212215608886" style="zoom:80%;" />



但最后效果相比之下依然不如randn的理想，故若这里使用正态分布初始化最好还是使方差小一点（参数一开始不要差距太大）**恩达：参数接近0，不相同。 沐：方差0.01**

这里只是为了试试参数波动大会怎么样，导致需要更多的epoch来减小loss，这是件很贵的事情

#### 初始化参数时使其全部为1





<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212221216784.png" alt="image-20211212221216784" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212221311935.png" alt="image-20211212221311935" style="zoom:67%;" />



<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212221229038.png" alt="image-20211212221229038" style="zoom: 80%;" />

通过参数矩阵可以看出，正如吴恩达所说，gradient descent每次对同一个layer的改变都一样，导致每一层的感知机们都一模一样



同样的 参数全初始化为0的时候，训练结束后更有趣

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211212223019374.png" alt="image-20211212223019374" style="zoom:67%;" />



## 多层感知机的简洁实现

通过高级API更简洁地实现多层感知机

```python
import torch
from torch import nn
from d2l import torch as d2l
```

与softmax回归的简洁实现相⽐，唯⼀的区别是我们添加了2个全连接层（之前我们只添加了1个全连接层）。第⼀层是隐藏层，它包含256个隐藏单元，并使⽤ReLU激活函数。第⼆层是输出层

```python
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784,256),
                    nn.ReLU(),
                    nn.Linear(256,10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        
net.apply(init_weights)
```

训练过程的实现与我们实现softmax回归时完全相同，这种模块化设计使我们能够将与和模型架构有关的内容独⽴出来

```python
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
```

```python
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```



## QA环节

Q：MLP vs SVM？

A：SVM替代了感知机，多层感知机解决了XOR的问题，但没有流行是因为

1. MLP有很多超参数，相比之下SVM不需要调太多超参数。 
2. SVM有很漂亮的数学理论，学术界更认可，即使二者效果可能差不多

不过MLP比较方便转其他模型，如CNN,RNN,transformer等



Q：增加每层感知机个数和增加层数有什么区别？

A：增加层数或者增加每层的感知机个数都可以增加模型的复杂度。但如果一层有太多的感知机，模型不好训练（容易overfitting）。故深度学习的`深`就是指层数深（理论上可能是因为模型可以一步一步地学）



Q：ReLU为什么管用？在大于0的时候不也是线性的么？

A：ReLU不是线性函数，线性函数一定是严格的一条线。故ReLU能够打乱线性性（激活函数的本质）

沐：激活函数对模型的影响大小远远没有隐藏层大小等超参数重要，故一般ReLU即可



Q：最终训练出神经网络的参数可以是动态的吗？训练完后参数必须是死的吗？

A：训练完后的模型参数是不变的，动态会出问题，这样对于同样的输入可能会有不同的输出（Goole黑猩猩事件）



# ==模型选择、过拟合和欠拟合==

## 两种误差

+ **训练误差**：模型在训练数据上的误差
+ **泛化误差**：模型在新数据上的误差

eg：通过模拟考试来预测未来考试分数

学生A通过背答案在模考中拿到了好成绩，学生B知道答案后面的原因：那么A模考成绩甚至可能比B好，但真实考试准拉胯



## 验证数据集和测试数据集

**验证数据集*（validation set）*：一个用来评估模型好坏的数据集，也就是之前说的测试集，==验证数据集一定不要跟训练数据混在一起==（常犯错误） 

**测试数据集**（test set）：只用一次的数据集（不能用它来调超参数），eg：未来的考试，我出价的房子的实际成交价，用在kaggle私有排行榜中的数据集



## K-折交叉验证(K-fold Cross Validation)



**在没有足够多数据时使用**（这是常态）

数据集大了就不用了，本来相对来说也不缺数据，而且训练成本太高

算法：

+ 将数据分割成K块
+ 从1到K轮流将每一块数据作为验证集，其余作为训练集，进行K次训练
+ 报告K个验证误差的平均

常用 K=5或10



## 过拟合和欠拟合

模型容量需要匹配数据复杂度，否则可能导致过拟合和和欠拟合



**模型容量**：模型拟合各种函数的能力

+ 低容量的模型难以拟合训练数据（underfitting）
+ 高容量的模型可以记住所有训练数据（overfitting）

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211214172646701.png" alt="image-20211214172646701" style="zoom: 33%;" />

泛化误差和训练误差之间的gap通常用来衡量过拟合和欠拟合

过拟合本质上不是一件坏事情，模型首先要足够大，在足够大的前提下再去降低泛化误差，这是整个深度学习的核心

 

**估计模型容量**

不同种类算法之间难以比较，例如树模型和神经网络

给定一个模型，将有两个主要因素：1.参数的个数  2.参数值的选择范围



**VC维**

统计学习理论的核心思想，即对于一个分类模型，VC等于一个最大数据集的大小，不管如何给定标号，都存在一个模型来对它进行完美分类

**也就是一个模型能记住的最大的数据集维度**

例如，线性分类器的VC维是3，即3是其不管如何给定标号可以进行完美分类最大数据集大小

故支持N维输入的感知机的VC维是N+1

一些多层感知机的VC维是O (N log~2~N)



VC维提供了为什么一个模型好的理论依据，它可以衡量训练误差和泛化误差之间的间隔，但深度学习中很少使用，因为它衡量不是很准确且计算深度学习模型的VC维很困难



**数据复杂度**

多因素：样本个数、每个样本的元素的个数（例如一张图片的大小）、时间空间结构、多样性（多少类）



## 通过代码直观感受

通过多项式拟合来交互地探索这些概念

```python
import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
```

使用以下三阶多项式来生成训练和测试数据的标签

<img src="https://www.zhihu.com/equation?tex=y = 5 + 1.2x -3.4\frac{x^2}{2!}+5.6\frac{x^3}{3!}+\epsilon \quad where\,\epsilon {\text ~} N(0, 0.1^2)
" alt="y = 5 + 1.2x -3.4\frac{x^2}{2!}+5.6\frac{x^3}{3!}+\epsilon \quad where\,\epsilon {\text ~} N(0, 0.1^2)
" class="ee_img tr_noresize" eeimg="1">


```python
import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
```

使用三阶多项式来生成训练和测试数据的标签

```python
max_degree = 20
n_train, n_test = 100, 100
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

# 生成 (n_train+n_test)×1 大小的矩阵
features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
# 列向量features 通过 广播机制与 行向量np.arange(max_degree).reshape(1, -1) 进行元素的次方运算
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
# 除上阶乘（防止过大）
for i in range(max_degree):
    poly_features[:,i] /= math.gamma(i+1)  # woc伽马函数！！！即(n-1)！
# 乘上系数 通过类似内积运算相加（每个样本的前五项与true_w做内积所得向量）
labels = np.dot(poly_features, true_w)
# 加上偏移量
labels += np.random.normal(scale=0.1, size=labels.shape) 
```

numpy ndarry 转 tensor

```python
# numpy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(nep, dtype=d2l.float32)
                                           for nep in [true_w, features,poly_features, labels]]
features[:2], poly_features[:2,:], labels[:2]  #poly_features[:2]也行
```

实现评估模型损失的函数

```python
def evaluate_loss(net, data_iter, loss):
    '''评估给定数据集上模型的损失'''
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]
```

对模型进行训练和测试

```python
def train(train_features, test_features, train_labels, test_labels, 
          num_epochs=400):
    loss = nn.MSELoss()
    
    # shape[-1]表示读取最后一个维度，这里也就是输入的维度
    input_shape = train_features.shape[-1]
    # 单层线性回归 不设置偏置，因为之前已经设置了
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)),
                              batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1,num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        # 每20个epoch我们在图上记录一次
        if epoch == 0 or (epoch + 1) % 20 ==0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                   evaluate_loss(net, test_iter, loss )))
    print('weight', net[0].weight.data.numpy())
    
```

**使用前4项进行模型训练**（和之前的模型匹配）

```python
train(train_features=poly_features[:n_train, :4], test_features=poly_features[n_train:, :4],
     train_labels=labels[:n_train], test_labels=labels[n_train:])
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211214211211181.png" alt="image-20211214211211181" style="zoom:80%;" />

有时也会出现如下情况

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211214211150755.png" alt="image-20211214211150755" style="zoom:80%;" />

结果还行蛮不错，毕竟模型匹配了，为什么二者会有这么大区别呢？ 看了随机初始化确实是有一定影响的。*可以理解谷歌为什么对几个同样模型进行不同的随机初始化后再训练，然后再根据这些模型的预测求平均*



**使用前2项进行模型训练**

```python
train(train_features=poly_features[:n_train, :2], test_features=poly_features[n_train:, :2],
     train_labels=labels[:n_train], test_labels=labels[n_train:])
```



<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211214213117216.png" alt="image-20211214213117216" style="zoom:80%;" />

==underfitting==



**使用高阶多项式函数拟合**

```python
train(train_features=poly_features[:n_train, :], test_features=poly_features[n_train:, :],
     train_labels=labels[:n_train], test_labels=labels[n_train:])
```

涅普！意外的还不错？

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211214213604674.png" alt="image-20211214213604674" style="zoom:80%;" />

再来一次

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211214213743351.png" alt="image-20211214213743351" style="zoom: 80%;" />

==overfitting？==

可以看见该模型把后面许多不该学的参数也学进去了



### Kang脑洞一下(只能说思路之混乱)


<img src="https://www.zhihu.com/equation?tex=x ,\frac{x^2}{2!},\frac{x^3}{3!},...$$ )

是不是初始化样本的时候x太接近0了，于是增加初始化正态分布的均值，发现若x过大会出现NAN，其他情况依然系数相差不大，如图为 均值=3 的情况，此时系数依然很接近真实值

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211214210035205.png" alt="image-20211214210035205" style="zoom:80%;" />

分析

1. ~~可能是因为公式中的阶乘抵消了一部分~~

2. ~~可能是因为初始化样本时正态分布方差太小，故样本中x相对集中，尝试增大方差~~

   `features = np.random.normal(0, 2,size=(n_train + n_test, 1))`

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211214210119644.png" alt="image-20211214210119644" style="zoom:80%;" />

3. ~~有没有可能，`net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))`中的模型定义就是：~~

" alt="x ,\frac{x^2}{2!},\frac{x^3}{3!},...$$ )

是不是初始化样本的时候x太接近0了，于是增加初始化正态分布的均值，发现若x过大会出现NAN，其他情况依然系数相差不大，如图为 均值=3 的情况，此时系数依然很接近真实值

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211214210035205.png" alt="image-20211214210035205" style="zoom:80%;" />

分析

1. ~~可能是因为公式中的阶乘抵消了一部分~~

2. ~~可能是因为初始化样本时正态分布方差太小，故样本中x相对集中，尝试增大方差~~

   `features = np.random.normal(0, 2,size=(n_train + n_test, 1))`

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211214210119644.png" alt="image-20211214210119644" style="zoom:80%;" />

3. ~~有没有可能，`net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))`中的模型定义就是：~~

" class="ee_img tr_noresize" eeimg="1">
y = \theta_0 + \theta_1x -\theta_2\frac{x^2}{2!}+\theta_3\frac{x^3}{3!}+ ...

<img src="https://www.zhihu.com/equation?tex=4. **还有一种可能，自己没把问题搞清楚就开始乱整乱猜了！！！**

我们传入数据集的数据是`poly_features`，在这里，我们实际上是将
" alt="4. **还有一种可能，自己没把问题搞清楚就开始乱整乱猜了！！！**

我们传入数据集的数据是`poly_features`，在这里，我们实际上是将
" class="ee_img tr_noresize" eeimg="1">
x ,\frac{x^2}{2!},\frac{x^3}{3!},...

<img src="https://www.zhihu.com/equation?tex=分别作为输入，使其进入线性模型！！！

==总结！！！==

1. 对于单变量的Polynome回归，可以用多变量的线性回归去拟合（每一个自变量对应多项式的一项）

2. 可以用此法进行泰勒展开来拟合各种函数



## QA环节

Q：SVM和神经网络相比有什么缺点？

N：SVM使用一个kernel来进行模型匹配，算起来很难，例如对于百万千万的数据集，SVM难以求解，而神经网络可以通过梯度下降来求解。其次SVM的可调性不高，即参数对模型影响不是特别大。相比之下，Mu：神经网络的优点，==**神经网络本身就是一种语言**==，它很灵活，可编程性很高

Q：对于神经网络本身就是一种语言，可不可以理解为神经网络可以对万物建模，理论上可以拟合所有函数？

A：理论上单层神经网络就可以拟合所有函数，但实际上很难求解，这也是为什么会有CNN（进行空间信息的限制）、**RNN（进行时序信息的限制！）**等来尽量帮助神经网络进行训练

艺术（美但是不一定有用）、工程（有用）、科学（理解探索）

神经网络看上去像科学，但实际上更偏向工程，最开始是艺术（doge）



Q：对于时间序列的数据集如果有自相关性，应该怎么办？

A：对于时间序列的预测只能将前一段时间作为训练集，后面一段时间做测试集



Q：在做数据集的数据清洗和features scaling时，是否可以将训练集和测试集放在一起计算？

A：可以的，当然分开只计算训练集也可以，具体情况看实际应用



Q：K-fold Cross Validation   每块训练时获得的参数可能都不一样，最后模型的选择应该怎么办？

A：所有求平均（大数定律）

其实有很多做法：

1. 通过K折交叉验证确定理想的超参数，然后再在整个训练集上重新训练（最常见）
2. 不再重新训练，就随便找K折里面的一折或者找精度最好的一折
3. 把K个模型全部拿下来，做预测的时候，使用K个模型做预测，再求均值（更具有稳定性但实际验证的时候代价为K倍）



Q：如何有效设计超参数，是不是只能搜索？最好用的搜索是贝叶斯方法还是网格、随机？

A：超参数的选择方式是指数级的（例如学习率有3种选择，层数有3种选择，每层个数有3种选择，就3×3×3）。网格：遍历所有选择；随机：就。。随机呗？Mu：主要还是靠自己设定（老中医）



Q：假设样本的类型很不平衡，也就是有的类别样本很多，有的类别很少，应该怎么办？

A：如果数据集不大的话，那么验证数据集最后两类数目差不多，这样防止分类器过度判多的那一类而忽略少的那一类

注：如果真实世界的类型本来就不平衡，那么就可以直接搞，但是如果是因为采样没采好，那就需要进行处理：复制小样本样本（宏毅：狗狗对称一下，缩放一下等等）或者在loss中增加小样本的权重



Q：随着迭代次数增加，验证loss先下降后上升，是因为过拟合吗？

A：是的



# ==权重衰减(weight decay)==

*为了帮孩子减小噪音影响的老母亲（bushi）*



通过限制参数值的选择返回来控制模型容量（硬性限制，都得给我小）
" alt="分别作为输入，使其进入线性模型！！！

==总结！！！==

1. 对于单变量的Polynome回归，可以用多变量的线性回归去拟合（每一个自变量对应多项式的一项）

2. 可以用此法进行泰勒展开来拟合各种函数



## QA环节

Q：SVM和神经网络相比有什么缺点？

N：SVM使用一个kernel来进行模型匹配，算起来很难，例如对于百万千万的数据集，SVM难以求解，而神经网络可以通过梯度下降来求解。其次SVM的可调性不高，即参数对模型影响不是特别大。相比之下，Mu：神经网络的优点，==**神经网络本身就是一种语言**==，它很灵活，可编程性很高

Q：对于神经网络本身就是一种语言，可不可以理解为神经网络可以对万物建模，理论上可以拟合所有函数？

A：理论上单层神经网络就可以拟合所有函数，但实际上很难求解，这也是为什么会有CNN（进行空间信息的限制）、**RNN（进行时序信息的限制！）**等来尽量帮助神经网络进行训练

艺术（美但是不一定有用）、工程（有用）、科学（理解探索）

神经网络看上去像科学，但实际上更偏向工程，最开始是艺术（doge）



Q：对于时间序列的数据集如果有自相关性，应该怎么办？

A：对于时间序列的预测只能将前一段时间作为训练集，后面一段时间做测试集



Q：在做数据集的数据清洗和features scaling时，是否可以将训练集和测试集放在一起计算？

A：可以的，当然分开只计算训练集也可以，具体情况看实际应用



Q：K-fold Cross Validation   每块训练时获得的参数可能都不一样，最后模型的选择应该怎么办？

A：所有求平均（大数定律）

其实有很多做法：

1. 通过K折交叉验证确定理想的超参数，然后再在整个训练集上重新训练（最常见）
2. 不再重新训练，就随便找K折里面的一折或者找精度最好的一折
3. 把K个模型全部拿下来，做预测的时候，使用K个模型做预测，再求均值（更具有稳定性但实际验证的时候代价为K倍）



Q：如何有效设计超参数，是不是只能搜索？最好用的搜索是贝叶斯方法还是网格、随机？

A：超参数的选择方式是指数级的（例如学习率有3种选择，层数有3种选择，每层个数有3种选择，就3×3×3）。网格：遍历所有选择；随机：就。。随机呗？Mu：主要还是靠自己设定（老中医）



Q：假设样本的类型很不平衡，也就是有的类别样本很多，有的类别很少，应该怎么办？

A：如果数据集不大的话，那么验证数据集最后两类数目差不多，这样防止分类器过度判多的那一类而忽略少的那一类

注：如果真实世界的类型本来就不平衡，那么就可以直接搞，但是如果是因为采样没采好，那就需要进行处理：复制小样本样本（宏毅：狗狗对称一下，缩放一下等等）或者在loss中增加小样本的权重



Q：随着迭代次数增加，验证loss先下降后上升，是因为过拟合吗？

A：是的



# ==权重衰减(weight decay)==

*为了帮孩子减小噪音影响的老母亲（bushi）*



通过限制参数值的选择返回来控制模型容量（硬性限制，都得给我小）
" class="ee_img tr_noresize" eeimg="1">
\min\,l({\bf w,b}) \quad \text {subject to} ||{\bf w}||^2 \le \theta

<img src="https://www.zhihu.com/equation?tex=通常不限制偏移b，限不限制都差不多（恩达也说了）

θ越小意味着正则化越强



**使用均方范数作为柔性限制**（更平滑，就像Photoshop拉曲线一样）

对于每个θ，都可以找到λ使得之前的目标函数等价于
" alt="通常不限制偏移b，限不限制都差不多（恩达也说了）

θ越小意味着正则化越强



**使用均方范数作为柔性限制**（更平滑，就像Photoshop拉曲线一样）

对于每个θ，都可以找到λ使得之前的目标函数等价于
" class="ee_img tr_noresize" eeimg="1">
\min\,l({\bf w,b}) + \frac \lambda 2||{\bf w}||^2

<img src="https://www.zhihu.com/equation?tex=可以通过拉格朗日乘子来证明

**超参数λ**控制了正则项的重要程度：λ=0，无作用； λ→∞， **w*** → **0**



直观理解，实际上就是loss和penalty的权衡，相当于没有penalty，penalty将参数值向原点拉（这个过程punish减小，loss增加），直到punish的减小和loss的增加相平衡。

如图，绿色等高线为loss，其中心为loss最小。黄色等高线为penalty。**w***为loss和penalty的平衡点

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215114548740.png" alt="image-20211215114548740" style="zoom: 67%;" />



参数更新新法则（跟恩达一样）

计算梯度
" alt="可以通过拉格朗日乘子来证明

**超参数λ**控制了正则项的重要程度：λ=0，无作用； λ→∞， **w*** → **0**



直观理解，实际上就是loss和penalty的权衡，相当于没有penalty，penalty将参数值向原点拉（这个过程punish减小，loss增加），直到punish的减小和loss的增加相平衡。

如图，绿色等高线为loss，其中心为loss最小。黄色等高线为penalty。**w***为loss和penalty的平衡点

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215114548740.png" alt="image-20211215114548740" style="zoom: 67%;" />



参数更新新法则（跟恩达一样）

计算梯度
" class="ee_img tr_noresize" eeimg="1">
\frac \part {\part {\bf w}}\left(l({\bf w, b})+\frac \lambda 2 ||{\bf w}||^2\right) = \frac {\part l{\bf(w, b)}} {\part {\bf w}} + \lambda {\bf w}

<img src="https://www.zhihu.com/equation?tex=更新参数
" alt="更新参数
" class="ee_img tr_noresize" eeimg="1">
{\bf w}_{t+1} = (1-\alpha\lambda){\bf w}_t - \alpha \frac {\part l{\bf(w_t, b_t)}} {\part {\bf w_t}}

<img src="https://www.zhihu.com/equation?tex=很明显由公式可知，因为λ的引入，相当于在更新前就先对所有参数进行一次减小，故叫权重衰退（又喜提超参数λ）



## 从零开始代码实现

```python
%matplotlib inline
import torch
from torch import nn
from d2l import torch as d2l
```

生成人工数据集
" alt="很明显由公式可知，因为λ的引入，相当于在更新前就先对所有参数进行一次减小，故叫权重衰退（又喜提超参数λ）



## 从零开始代码实现

```python
%matplotlib inline
import torch
from torch import nn
from d2l import torch as d2l
```

生成人工数据集
" class="ee_img tr_noresize" eeimg="1">
y = 0.05 + \sum_{i=1}^d 0.01x_i+\epsilon \quad where \, \epsilon {\text ~} N(0, 0.01^2)

<img src="https://www.zhihu.com/equation?tex=```python
# 刻意让训练数据集小， num_inputs也就是公式里面的i啦~ 
# 这里训练集小，特征很大，故很容易过拟合
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
```

初始化模型参数

```python
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1,requires_grad=True)
    return [w,b]  # 返回值是一个list，0维度对应w，1维度对应b
```

定义L2范数惩罚

```python
# 这里没有λ,因为要调它（doge）
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
```

```python
定义训练函数
```

```python
def train(lambd):
    #初始化模型参数
    w, b = init_params()
    # lambda为匿名函数，也就是你给我一个X，我给你代到线性模型里面算
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                           xlim=[5, num_epochs], legend=['train','test'])
    for epoch in range(num_epochs):   # 每一个epoch
        for X, y in train_iter:        # 每一个batch
            with torch.enable_grad():  # 新版本可注释
                # 增加L2范数惩罚项，广播机制，运算时把l2_penalty(w)看做一个长度为batch_size的向量
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward() # l是个向量，先求和再backward()
            d2l.sgd([w, b], lr, batch_size) # 进行梯度下降
        if(epoch + 1) % 5 == 0:
            animator.add(epoch+1, (d2l.evaluate_loss(net, train_iter, loss),
                                   d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：',torch.norm(w).item())
```



首先忽略正则化直接训练
我们现在⽤lambd = 0禁⽤权重衰减后运⾏这个代码。注意，这⾥训练误差有了减少，但测试误差没有减少。这意味着出现了严重的过拟合。这是过拟合的⼀个典型例⼦。
训练集小，特征又多，故只需要尽可能把训练集背下来，然后一到测试集准拉胯

```python
train(lambd=0)
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215130806149.png" alt="image-20211215130806149" style="zoom:80%;" />

接下来增加正则项，减小参数

```python
train(lambd=3)
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215130848290.png" alt="image-20211215130848290" style="zoom:80%;" />

可以看见迭代50轮之后train loss几乎不变，因为罚对其参数进行了限制，使权重不能过大



## 动手一下(使用L1范数作为正则化项)

即修改一下penalty（图中输出的是L1范数）

```python
def l1_penalty(w):
    return torch.sum(torch.abs(w))
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215191342131.png" alt="image-20211215191342131" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215191401917.png" alt="image-20211215191401917" style="zoom:80%;" />

6666666，没想到L1范数在这个问题中效果出乎意料的好！！！**==动手学习深度学习==** 自己要多试一试！



**Kang猜测原因**：可能是此问题汇总因为L1正则化项对模型的限制更强，由于初始化的参数都比较小，所以L2求平方后反而更小了。故可能导致梯度也比较小，而L1是绝对值求和，故相对来说会大一些（有更多和loss讨价还价的资本）

验证一下初始化的系数矩阵，确实很有可能是这个原因！

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215192713827.png" alt="image-20211215192713827" style="zoom: 80%;" />



## 简洁实现

其实`torch.optim.SGD()`中提供了对参数的权重衰减

```python
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        # 初始化参数
        param.data.normal_()
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    # bias没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay':wd},   #按照wd对权重进行weight_decay
        {"params":net[0].bias}],lr =lr)
    
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                           xlim=[5,num_epochs], legend=['train', 'test'])
    
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
        	l.backward()
        	trainer.step()
        
        # 设置每隔多少个epoch在图像上记录一次，在这个问题中，画图的代价反而比计算高很多（不画的话秒出结果）
        if(epoch + 1) % 1 == 0:
            animator.add(epoch+1, (d2l.evaluate_loss(net, train_iter, loss),
                                   d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数', net[0].weight.norm().item())
```

最开始kang一看 “我靠咋没有画曲线！”？	玩了半个小时的找不同游戏后，发现原来是将最后的`if(epoch + 1) % 1 == 0:`语句少了个缩进，导致其没在for循环，也就相当于只画了一个点

然后一看 我靠为什么加入权重衰退`wd=3`时最后算的范数和之前差距很大？零点几和5点几的差距？	原来是以上代码中 `l.backward()`和`trainer.step()`也少了缩进，导致是每个epoch进行一次gradient descent而不是每个batch进行一次！

今天的最大感悟：==python是一门缩进敏感的语言==



<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215184154522.png" alt="image-20211215184154522" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215184201486.png" alt="image-20211215184201486" style="zoom:80%;" />

## QA环节

Q：请问现在pytorch是否支持复数神经网络（也就是nn输入输出权重激活函数都是复数，loss则是一个复数到实数的映射）？   （我靠脑洞这么大的吗！！！）

A：应该是不支持的，但是也可以实现功能，因为复数实际上就是把一个数变成两位，故可以加一维来实现效果



Q：为什么参数小复杂度就低？

A：并不是参数小复杂度就低，而是通过正则化项来限制参数的取值范围，参数取值范围小了，复杂度就降低了



Q：实际的权重衰减值一般设置多少为好呢？为什么之前跑代码的时候感觉权重衰退效果并不是那么好？

A：一般是取e-1，e-2，e-3，e-4。如果模型很复杂的话权重衰减并不会带来很好的效果，



Q：为什么要用权重衰退要把w往小拉？万一需要的w参数本来就需要很大那不就适得其反了吗？

A：（kang：要是这样的话loss就更有话语权了）；Mu：我们的算法总是会尝试记住数据，因此会学到很多噪音（也正是因为有噪音，才需要权重衰减拉他一把），所以参数一般都会偏大需要权重衰减将其往回拉（λ太小了就拉的不够，λ太大就过了，哎，正则化项就跟到老母亲一样，不容易啊）



# ==丢弃法(Dropput)==

**动机**：一个好的模型需要对输入数据的扰动鲁棒

使用有噪音的数据等价于Tikhonov正则

丢弃法：在**全连接层**之间加入噪音（训练时每次丢弃一些神经元，实际上就是把对应神经元的输出置零）（故丢弃法实际上是一种正则，实际意义和正则差不多）



**Dropout定义：无偏加入噪音**

对**x**加入噪音得到**x'**，我们希望
" alt="```python
# 刻意让训练数据集小， num_inputs也就是公式里面的i啦~ 
# 这里训练集小，特征很大，故很容易过拟合
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
```

初始化模型参数

```python
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1,requires_grad=True)
    return [w,b]  # 返回值是一个list，0维度对应w，1维度对应b
```

定义L2范数惩罚

```python
# 这里没有λ,因为要调它（doge）
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
```

```python
定义训练函数
```

```python
def train(lambd):
    #初始化模型参数
    w, b = init_params()
    # lambda为匿名函数，也就是你给我一个X，我给你代到线性模型里面算
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                           xlim=[5, num_epochs], legend=['train','test'])
    for epoch in range(num_epochs):   # 每一个epoch
        for X, y in train_iter:        # 每一个batch
            with torch.enable_grad():  # 新版本可注释
                # 增加L2范数惩罚项，广播机制，运算时把l2_penalty(w)看做一个长度为batch_size的向量
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward() # l是个向量，先求和再backward()
            d2l.sgd([w, b], lr, batch_size) # 进行梯度下降
        if(epoch + 1) % 5 == 0:
            animator.add(epoch+1, (d2l.evaluate_loss(net, train_iter, loss),
                                   d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：',torch.norm(w).item())
```



首先忽略正则化直接训练
我们现在⽤lambd = 0禁⽤权重衰减后运⾏这个代码。注意，这⾥训练误差有了减少，但测试误差没有减少。这意味着出现了严重的过拟合。这是过拟合的⼀个典型例⼦。
训练集小，特征又多，故只需要尽可能把训练集背下来，然后一到测试集准拉胯

```python
train(lambd=0)
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215130806149.png" alt="image-20211215130806149" style="zoom:80%;" />

接下来增加正则项，减小参数

```python
train(lambd=3)
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215130848290.png" alt="image-20211215130848290" style="zoom:80%;" />

可以看见迭代50轮之后train loss几乎不变，因为罚对其参数进行了限制，使权重不能过大



## 动手一下(使用L1范数作为正则化项)

即修改一下penalty（图中输出的是L1范数）

```python
def l1_penalty(w):
    return torch.sum(torch.abs(w))
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215191342131.png" alt="image-20211215191342131" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215191401917.png" alt="image-20211215191401917" style="zoom:80%;" />

6666666，没想到L1范数在这个问题中效果出乎意料的好！！！**==动手学习深度学习==** 自己要多试一试！



**Kang猜测原因**：可能是此问题汇总因为L1正则化项对模型的限制更强，由于初始化的参数都比较小，所以L2求平方后反而更小了。故可能导致梯度也比较小，而L1是绝对值求和，故相对来说会大一些（有更多和loss讨价还价的资本）

验证一下初始化的系数矩阵，确实很有可能是这个原因！

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215192713827.png" alt="image-20211215192713827" style="zoom: 80%;" />



## 简洁实现

其实`torch.optim.SGD()`中提供了对参数的权重衰减

```python
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        # 初始化参数
        param.data.normal_()
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    # bias没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay':wd},   #按照wd对权重进行weight_decay
        {"params":net[0].bias}],lr =lr)
    
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                           xlim=[5,num_epochs], legend=['train', 'test'])
    
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
        	l.backward()
        	trainer.step()
        
        # 设置每隔多少个epoch在图像上记录一次，在这个问题中，画图的代价反而比计算高很多（不画的话秒出结果）
        if(epoch + 1) % 1 == 0:
            animator.add(epoch+1, (d2l.evaluate_loss(net, train_iter, loss),
                                   d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数', net[0].weight.norm().item())
```

最开始kang一看 “我靠咋没有画曲线！”？	玩了半个小时的找不同游戏后，发现原来是将最后的`if(epoch + 1) % 1 == 0:`语句少了个缩进，导致其没在for循环，也就相当于只画了一个点

然后一看 我靠为什么加入权重衰退`wd=3`时最后算的范数和之前差距很大？零点几和5点几的差距？	原来是以上代码中 `l.backward()`和`trainer.step()`也少了缩进，导致是每个epoch进行一次gradient descent而不是每个batch进行一次！

今天的最大感悟：==python是一门缩进敏感的语言==



<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215184154522.png" alt="image-20211215184154522" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215184201486.png" alt="image-20211215184201486" style="zoom:80%;" />

## QA环节

Q：请问现在pytorch是否支持复数神经网络（也就是nn输入输出权重激活函数都是复数，loss则是一个复数到实数的映射）？   （我靠脑洞这么大的吗！！！）

A：应该是不支持的，但是也可以实现功能，因为复数实际上就是把一个数变成两位，故可以加一维来实现效果



Q：为什么参数小复杂度就低？

A：并不是参数小复杂度就低，而是通过正则化项来限制参数的取值范围，参数取值范围小了，复杂度就降低了



Q：实际的权重衰减值一般设置多少为好呢？为什么之前跑代码的时候感觉权重衰退效果并不是那么好？

A：一般是取e-1，e-2，e-3，e-4。如果模型很复杂的话权重衰减并不会带来很好的效果，



Q：为什么要用权重衰退要把w往小拉？万一需要的w参数本来就需要很大那不就适得其反了吗？

A：（kang：要是这样的话loss就更有话语权了）；Mu：我们的算法总是会尝试记住数据，因此会学到很多噪音（也正是因为有噪音，才需要权重衰减拉他一把），所以参数一般都会偏大需要权重衰减将其往回拉（λ太小了就拉的不够，λ太大就过了，哎，正则化项就跟到老母亲一样，不容易啊）



# ==丢弃法(Dropput)==

**动机**：一个好的模型需要对输入数据的扰动鲁棒

使用有噪音的数据等价于Tikhonov正则

丢弃法：在**全连接层**之间加入噪音（训练时每次丢弃一些神经元，实际上就是把对应神经元的输出置零）（故丢弃法实际上是一种正则，实际意义和正则差不多）



**Dropout定义：无偏加入噪音**

对**x**加入噪音得到**x'**，我们希望
" class="ee_img tr_noresize" eeimg="1">
E({\bf x'}) = {\bf x}

<img src="https://www.zhihu.com/equation?tex=丢弃法对每个元素进行如下扰动
" alt="丢弃法对每个元素进行如下扰动
" class="ee_img tr_noresize" eeimg="1">
x_i'=
\begin{cases}

0 \quad &{\text with \,\,probablity\,\,p}\\
\frac {x_i} {1-p} &{\text otherwise}

\end{cases}

<img src="https://www.zhihu.com/equation?tex=完毕！

训练的时候

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215212456118.png" alt="image-20211215212456118" style="zoom: 50%;" />

推理的时候，不需要Dropout，所有的正则都只是在训练中使用，推理中不使用

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215215540262.png" alt="image-20211215215540262" style="zoom:50%;" />



总结

+ 丢弃法将一些输出项随机置0来控制模型复杂度（**随机选择全连接层一些权重进行更新**）
+ 常作用在多层感知机的隐藏层输出上
+ 丢弃概率p是控制模型复杂度的超参数



## 丢弃法实现

我们实现 `dropout_layer` 函数，该函数以dropout的概率丢弃张量输⼊X中的元素，如上所述重新缩放剩余部分：将剩余部分除以1.0-dropout

```python
import torch
from torch import nn
from d2l import torch as d2l

def dropout_layer(X, dropout):
    # dropout必须是概率
    assert 0 <= dropout <=1
    # dropput为1 全丢
    if dropout == 1:
        return torch.zeros_like(X)
    # dropout为0 没变
    if dropout == 0:
        return X
    # 妙啊！X.shape的向量每个值取0-1的均匀随机分布，判断与dropout的大小（bool），再转为float（0,1.0）
    mask = (torch.randn(X.shape) > dropout).float()
    # 这里直接做乘法而不是按对应下标去选元素，相对来说做乘法远比选元素快（比如对GPU来说）
    return mask * X / (1.0 - dropout)
```

```python
# test
X = torch.arange(16, dtype=torch.float32).reshape((2,8))
print(X)
print(dropout_layer(X, 0))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1))
```

![image-20211216004824837](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211216004824837.png)



定义具有两个隐藏层的多层感知机，每个隐藏层有256个单元

```python
# 经典咏流传
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

dropout1, dropout2 = 0.2, 0.5

# Kang后来盲猜class Nep(nn.Module): 意思是Nep继承nn.Moudle!!! 芜湖果然！
class Nep(nn.Module):
    # 如果is_training = True 的话就把这些参数传进来进行初始化
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Nep, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
    
    # forward是重写了方法么？ Yes！
    # 查官方文档：torch.nn.Module.forward(*input)
    # Defines the computation performed at every call.
    # Should be overridden by all subclasses.
    def forward(self, X):
        # 把 X reshape后传到第一个隐藏层，再进行relu， 得到H1
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 训练模式就进行dropout
        if self.training == True:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2, dropout2)
        # 输出层就不做dropout了，否则正则性太强
        out = self.lin3(H2)
        return out

nep = Nep(num_inputs, num_outputs, num_hiddens1, num_hiddens2)       
```



训练和测试

```python
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(nep.parameters(), lr=lr)
d2l.train_ch3(nep, train_iter, test_iter, loss, num_epochs, trainer)
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211216004928352.png" alt="image-20211216004928352" style="zoom:80%;" />

相比之前单隐藏层的多层感知机，这里精度和loss都有所改善，但是很小。但是想一想，现在我们使用的是双隐藏层，故模型本身应该十分复杂，所以可见dropout对该模型起到的正则化效果。



## 简洁实现

```python
import torch
from torch import nn
import d2l.torch as d2l
```

直接在 `torch.nn.Sequential()` 中设置对应全连接层的dropout

```python
dropout1, dropout2 = 0.2, 0.5
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

net = nn.Sequential(nn.Flatten(),
                   nn.Linear(num_inputs, num_hiddens1),
                   nn.ReLU(),
                   # 在第一个全连接层后添加一个dropout层
                   nn.Dropout(dropout1),
                   nn.Linear(num_hiddens1, num_hiddens2),
                   nn.ReLU(),
                    # 在第个全连接层后添加一个dropout层
                   nn.Dropout(dropout2),
                   nn.Linear(num_hiddens2, num_outputs))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

对模型进行训练和测试

```python
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()  # 包含了softmax
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211216121438654.png" alt="image-20211216121438654" style="zoom:80%;" />



## QA环节

Q：dropout随机丢弃，如何保证正确性和可重复性？

A：神经网络没有正确性可言，只有精度好不好（从实际应用上）；对整个神经网络来说，随机性都挺大的，固没必要保证可重复性。如果实在想要dropout的丢弃有可重复性，可以固定随机种子

注：CUDA的cuDNN进行加速运算的时候也没有可重复性

没有必要追求可重复性，随机性本身就是一种平滑



Q：训练时使用dropout而推理时不用。会不会导致推理输出结果变化？（比如dropout=0.5，而推理输出时神经元的个数是训练时的两倍）

A：这就是为什么要把未置零的参数除以`1-dropout`，保证其放大后期望和未进行dropout时一样（Kang：可以理解为通过dropout训练完后的模型是将之前的所有dropout训练求了一个均值？而每一次dropout训练的期望都和原来一样，故求平均根据大数定律，整体还是一样）

直观理解就是dropout时将系数放大了，但神经元数目减少了；而推理时使用了所有神经元，但没有对系数进行放大。且二者期望相同



Q：dropout每次随机选择几个子网络，最后做平均的做法是不是类似于随机森林多决策树做投票的思想？

A：可以这么理解，但是dropout更像一个正则项



Q：dropout已经被Google申请专利，还可以用吗？

A：（doge）dropout、transformer、RNN。Google都帮你申请了。不过也可以用，律师没说不能用就行（doge）



Q：dropout和权重衰减都属于正则，为何dropout效果更好更常用呢？

A：实际上dropout没有weight decay常用。首先dropout只能对全连接层使用，weight decay可以在卷积、transformer使用。 相对来说dropout更好调参，因为更直观，决定就是丢多少



# ==数值稳定性==

“整个深度学习的工作都是在让数值更稳定”



（顺便再了解一下梯度下降和反向传播）

考虑如下有d层的神经网络
" alt="完毕！

训练的时候

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215212456118.png" alt="image-20211215212456118" style="zoom: 50%;" />

推理的时候，不需要Dropout，所有的正则都只是在训练中使用，推理中不使用

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211215215540262.png" alt="image-20211215215540262" style="zoom:50%;" />



总结

+ 丢弃法将一些输出项随机置0来控制模型复杂度（**随机选择全连接层一些权重进行更新**）
+ 常作用在多层感知机的隐藏层输出上
+ 丢弃概率p是控制模型复杂度的超参数



## 丢弃法实现

我们实现 `dropout_layer` 函数，该函数以dropout的概率丢弃张量输⼊X中的元素，如上所述重新缩放剩余部分：将剩余部分除以1.0-dropout

```python
import torch
from torch import nn
from d2l import torch as d2l

def dropout_layer(X, dropout):
    # dropout必须是概率
    assert 0 <= dropout <=1
    # dropput为1 全丢
    if dropout == 1:
        return torch.zeros_like(X)
    # dropout为0 没变
    if dropout == 0:
        return X
    # 妙啊！X.shape的向量每个值取0-1的均匀随机分布，判断与dropout的大小（bool），再转为float（0,1.0）
    mask = (torch.randn(X.shape) > dropout).float()
    # 这里直接做乘法而不是按对应下标去选元素，相对来说做乘法远比选元素快（比如对GPU来说）
    return mask * X / (1.0 - dropout)
```

```python
# test
X = torch.arange(16, dtype=torch.float32).reshape((2,8))
print(X)
print(dropout_layer(X, 0))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1))
```

![image-20211216004824837](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211216004824837.png)



定义具有两个隐藏层的多层感知机，每个隐藏层有256个单元

```python
# 经典咏流传
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

dropout1, dropout2 = 0.2, 0.5

# Kang后来盲猜class Nep(nn.Module): 意思是Nep继承nn.Moudle!!! 芜湖果然！
class Nep(nn.Module):
    # 如果is_training = True 的话就把这些参数传进来进行初始化
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Nep, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
    
    # forward是重写了方法么？ Yes！
    # 查官方文档：torch.nn.Module.forward(*input)
    # Defines the computation performed at every call.
    # Should be overridden by all subclasses.
    def forward(self, X):
        # 把 X reshape后传到第一个隐藏层，再进行relu， 得到H1
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 训练模式就进行dropout
        if self.training == True:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2, dropout2)
        # 输出层就不做dropout了，否则正则性太强
        out = self.lin3(H2)
        return out

nep = Nep(num_inputs, num_outputs, num_hiddens1, num_hiddens2)       
```



训练和测试

```python
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(nep.parameters(), lr=lr)
d2l.train_ch3(nep, train_iter, test_iter, loss, num_epochs, trainer)
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211216004928352.png" alt="image-20211216004928352" style="zoom:80%;" />

相比之前单隐藏层的多层感知机，这里精度和loss都有所改善，但是很小。但是想一想，现在我们使用的是双隐藏层，故模型本身应该十分复杂，所以可见dropout对该模型起到的正则化效果。



## 简洁实现

```python
import torch
from torch import nn
import d2l.torch as d2l
```

直接在 `torch.nn.Sequential()` 中设置对应全连接层的dropout

```python
dropout1, dropout2 = 0.2, 0.5
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

net = nn.Sequential(nn.Flatten(),
                   nn.Linear(num_inputs, num_hiddens1),
                   nn.ReLU(),
                   # 在第一个全连接层后添加一个dropout层
                   nn.Dropout(dropout1),
                   nn.Linear(num_hiddens1, num_hiddens2),
                   nn.ReLU(),
                    # 在第个全连接层后添加一个dropout层
                   nn.Dropout(dropout2),
                   nn.Linear(num_hiddens2, num_outputs))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

对模型进行训练和测试

```python
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()  # 包含了softmax
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211216121438654.png" alt="image-20211216121438654" style="zoom:80%;" />



## QA环节

Q：dropout随机丢弃，如何保证正确性和可重复性？

A：神经网络没有正确性可言，只有精度好不好（从实际应用上）；对整个神经网络来说，随机性都挺大的，固没必要保证可重复性。如果实在想要dropout的丢弃有可重复性，可以固定随机种子

注：CUDA的cuDNN进行加速运算的时候也没有可重复性

没有必要追求可重复性，随机性本身就是一种平滑



Q：训练时使用dropout而推理时不用。会不会导致推理输出结果变化？（比如dropout=0.5，而推理输出时神经元的个数是训练时的两倍）

A：这就是为什么要把未置零的参数除以`1-dropout`，保证其放大后期望和未进行dropout时一样（Kang：可以理解为通过dropout训练完后的模型是将之前的所有dropout训练求了一个均值？而每一次dropout训练的期望都和原来一样，故求平均根据大数定律，整体还是一样）

直观理解就是dropout时将系数放大了，但神经元数目减少了；而推理时使用了所有神经元，但没有对系数进行放大。且二者期望相同



Q：dropout每次随机选择几个子网络，最后做平均的做法是不是类似于随机森林多决策树做投票的思想？

A：可以这么理解，但是dropout更像一个正则项



Q：dropout已经被Google申请专利，还可以用吗？

A：（doge）dropout、transformer、RNN。Google都帮你申请了。不过也可以用，律师没说不能用就行（doge）



Q：dropout和权重衰减都属于正则，为何dropout效果更好更常用呢？

A：实际上dropout没有weight decay常用。首先dropout只能对全连接层使用，weight decay可以在卷积、transformer使用。 相对来说dropout更好调参，因为更直观，决定就是丢多少



# ==数值稳定性==

“整个深度学习的工作都是在让数值更稳定”



（顺便再了解一下梯度下降和反向传播）

考虑如下有d层的神经网络
" class="ee_img tr_noresize" eeimg="1">
{\text第t层的输出：} {\bf h}^t = f_t({\bf h}^{t-1})\\
{\text Loss}= l\circ f_d \circ ... \circ f_1({\bf x})

<img src="https://www.zhihu.com/equation?tex=计算损失 `l` 关于参数 **`W`**~t~ 的梯度，由链式法则：
" alt="计算损失 `l` 关于参数 **`W`**~t~ 的梯度，由链式法则：
" class="ee_img tr_noresize" eeimg="1">
\frac {\part l} {\part {\bf W}^t} = \frac {\part l} {\part {\bf h}^d} \frac {\part {\bf h}^d} {\part {\bf h}^{d-1}}...\frac {\part {\bf h}^{t+1}} {\part {\bf h}^t} \frac {\part {\bf h}^t} {\part {\bf W}^t}

<img src="https://www.zhihu.com/equation?tex=由于两个向量求导得到的是一个矩阵。故这里要进行 `d-t` 次矩阵乘法



于是就带来数值稳定性的问题,，假设有100层：

1. **梯度爆炸**：1.5^100^ ≈ 4 × 10^17^
2. **梯度消失**：0.8^100^ ≈ 2 × 10^-10^



例子：假如如下MLP（为了简单省略偏移）

由于激活函数 `σ` 是按元素（分别对应上一层每一个线性运算）进行运算，故求导后是一个对角矩阵
" alt="由于两个向量求导得到的是一个矩阵。故这里要进行 `d-t` 次矩阵乘法



于是就带来数值稳定性的问题,，假设有100层：

1. **梯度爆炸**：1.5^100^ ≈ 4 × 10^17^
2. **梯度消失**：0.8^100^ ≈ 2 × 10^-10^



例子：假如如下MLP（为了简单省略偏移）

由于激活函数 `σ` 是按元素（分别对应上一层每一个线性运算）进行运算，故求导后是一个对角矩阵
" class="ee_img tr_noresize" eeimg="1">
{\text 第t层的输出：}{\bf h}_t=\sigma({\bf W}^t{\bf h}^{t-1})\\
\frac {\part {\bf h}^{t}} {\part {\bf h}^{t-1}} = diag(\sigma'({\bf W}^t{\bf h}^{t-1}))({\bf W}^t)^T

<img src="https://www.zhihu.com/equation?tex=之前的d-t次的矩阵乘法就是
" alt="之前的d-t次的矩阵乘法就是
" class="ee_img tr_noresize" eeimg="1">
\prod_{i=t}^{d-1}\frac {\part{\bf h}^{i+1}} {\part{\bf h}^{i}}=\prod_{i=t}^{d-1}diag(\sigma'({\bf W}^i{\bf h}^{i-1}))({\bf W}^i)^T

<img src="https://www.zhihu.com/equation?tex=## **梯度爆炸**

如果使用ReLU作为激活函数
" alt="## **梯度爆炸**

如果使用ReLU作为激活函数
" class="ee_img tr_noresize" eeimg="1">
\sigma(x) = \max(0, x)\quad and \quad \sigma'(x)=
\begin{cases}
1 &if\,x>0\\
0 &otherwise
\end{cases}

<img src="https://www.zhihu.com/equation?tex=故 $ <img src="https://www.zhihu.com/equation?tex=\prod_{i=t}^{d-1}\frac {\part{\bf h}^{i+1}} {\part{\bf h}^{i}}=\prod_{i=t}^{d-1}diag(\sigma'({\bf W}^i{\bf h}^{i-1}))({\bf W}^i)^T" alt="\prod_{i=t}^{d-1}\frac {\part{\bf h}^{i+1}} {\part{\bf h}^{i}}=\prod_{i=t}^{d-1}diag(\sigma'({\bf W}^i{\bf h}^{i-1}))({\bf W}^i)^T" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= 的一些元素会来自于 " alt=" 的一些元素会来自于 " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=\prod_{i=t}^{d-1}({\bf W}^i)^T" alt="\prod_{i=t}^{d-1}({\bf W}^i)^T" class="ee_img tr_noresize" eeimg="1"> $ ，另外一些元素是0，故若 `d-t` 很大，值也将很大，这就会导致梯度爆炸



梯度爆炸导致的问题

+ 值超出值域（infinity），尤其对于16位浮点数（数值区间 6e-5 - 6e4），尤其对于GPU来说使用的是float16
+ 对学习率敏感
  + 如果学习率大 -> 更大的参数值 -> 更大的梯度 -> 更大的参数值 -> ... Boom!
  + 如果学习率太小 -> 训练无进展
  + 故我们可能需要在训练过程不断调整学习率



## **梯度消失**

使用sigmoid作为激活函数
" alt="故 $ <img src="https://www.zhihu.com/equation?tex=\prod_{i=t}^{d-1}\frac {\part{\bf h}^{i+1}} {\part{\bf h}^{i}}=\prod_{i=t}^{d-1}diag(\sigma'({\bf W}^i{\bf h}^{i-1}))({\bf W}^i)^T" alt="\prod_{i=t}^{d-1}\frac {\part{\bf h}^{i+1}} {\part{\bf h}^{i}}=\prod_{i=t}^{d-1}diag(\sigma'({\bf W}^i{\bf h}^{i-1}))({\bf W}^i)^T" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= 的一些元素会来自于 " alt=" 的一些元素会来自于 " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=\prod_{i=t}^{d-1}({\bf W}^i)^T" alt="\prod_{i=t}^{d-1}({\bf W}^i)^T" class="ee_img tr_noresize" eeimg="1"> $ ，另外一些元素是0，故若 `d-t` 很大，值也将很大，这就会导致梯度爆炸



梯度爆炸导致的问题

+ 值超出值域（infinity），尤其对于16位浮点数（数值区间 6e-5 - 6e4），尤其对于GPU来说使用的是float16
+ 对学习率敏感
  + 如果学习率大 -> 更大的参数值 -> 更大的梯度 -> 更大的参数值 -> ... Boom!
  + 如果学习率太小 -> 训练无进展
  + 故我们可能需要在训练过程不断调整学习率



## **梯度消失**

使用sigmoid作为激活函数
" class="ee_img tr_noresize" eeimg="1">
\sigma(x) = \frac 1 {1+e^{-x}} \quad \sigma'(x)=\sigma(x)(1-\sigma(x))

<img src="https://www.zhihu.com/equation?tex=如果x绝对值比较大的话，sigmoid会接近0，故

$ <img src="https://www.zhihu.com/equation?tex=\prod_{i=t}^{d-1}\frac {\part{\bf h}^{i+1}} {\part{\bf h}^{i}}=\prod_{i=t}^{d-1}diag(\sigma'({\bf W}^i{\bf h}^{i-1}))({\bf W}^i)^T" alt="\prod_{i=t}^{d-1}\frac {\part{\bf h}^{i+1}} {\part{\bf h}^{i}}=\prod_{i=t}^{d-1}diag(\sigma'({\bf W}^i{\bf h}^{i-1}))({\bf W}^i)^T" class="ee_img tr_noresize" eeimg="1"> $ 的元素值可能是 `d-t` 个小数值的乘积



梯度消失导致的问题

+ 梯度值变为0，对 float16 尤为严重
+ 训练没有进展，无论怎么选择学习率
+ 对底部层（靠数据近的层）影响尤为严重（back propogation一步一步传回来后梯度越来越小），可能仅仅顶部层训练的好，无法让神经网络更深



故当数值过大或过小的时候都会导致数值问题，这常常发生在深度模型中，因为其会对n个数累乘



## 让训练更加稳定

目标：让梯度值在合理的范围内，如 [1e-6, 1e3]

核心思想：

+ 将乘法变加法：ResNet（CNN），LSTM（RNN）
+ 归一化：梯度归一化（例如转化为均值为0，方差为1），梯度剪裁（超出一定范围就强行截断）
+ **合理的权重初始 W 和激活函数 σ**



**权重初始化**：在合理值区间里随机初始参数，因为训练开始时更容易有数值不稳定（远离最优解的地方损失函数表面可能很复杂，梯度可能很大；最优解附近表面可能会比较平，梯度可能很小）。故像之前一样使用 N(0, 0.01) 来初始化可能对小网络没问题，但不能保证深度神经网络

**合理的权重初始 W**

将每层的每个元素的输出和梯度都看作随机变量，让它们的均值和方差都保持一致，即
" alt="如果x绝对值比较大的话，sigmoid会接近0，故

$ <img src="https://www.zhihu.com/equation?tex=\prod_{i=t}^{d-1}\frac {\part{\bf h}^{i+1}} {\part{\bf h}^{i}}=\prod_{i=t}^{d-1}diag(\sigma'({\bf W}^i{\bf h}^{i-1}))({\bf W}^i)^T" alt="\prod_{i=t}^{d-1}\frac {\part{\bf h}^{i+1}} {\part{\bf h}^{i}}=\prod_{i=t}^{d-1}diag(\sigma'({\bf W}^i{\bf h}^{i-1}))({\bf W}^i)^T" class="ee_img tr_noresize" eeimg="1"> $ 的元素值可能是 `d-t` 个小数值的乘积



梯度消失导致的问题

+ 梯度值变为0，对 float16 尤为严重
+ 训练没有进展，无论怎么选择学习率
+ 对底部层（靠数据近的层）影响尤为严重（back propogation一步一步传回来后梯度越来越小），可能仅仅顶部层训练的好，无法让神经网络更深



故当数值过大或过小的时候都会导致数值问题，这常常发生在深度模型中，因为其会对n个数累乘



## 让训练更加稳定

目标：让梯度值在合理的范围内，如 [1e-6, 1e3]

核心思想：

+ 将乘法变加法：ResNet（CNN），LSTM（RNN）
+ 归一化：梯度归一化（例如转化为均值为0，方差为1），梯度剪裁（超出一定范围就强行截断）
+ **合理的权重初始 W 和激活函数 σ**



**权重初始化**：在合理值区间里随机初始参数，因为训练开始时更容易有数值不稳定（远离最优解的地方损失函数表面可能很复杂，梯度可能很大；最优解附近表面可能会比较平，梯度可能很小）。故像之前一样使用 N(0, 0.01) 来初始化可能对小网络没问题，但不能保证深度神经网络

**合理的权重初始 W**

将每层的每个元素的输出和梯度都看作随机变量，让它们的均值和方差都保持一致，即
" class="ee_img tr_noresize" eeimg="1">
\forall i,j\\
{\text 正向：} E(h^t_i)=0 \quad Var(h^t_i) = a\\
{\text 反向：} E\left(\frac {\part l}{\part h_i^t}\right)=0 \quad Var\left(\frac {\part l}{\part h_i^t}\right) = b

<img src="https://www.zhihu.com/equation?tex=其中a，b都是常数

如何做到呢？

假设 1. 第t层的每一行每一列的权重 $ <img src="https://www.zhihu.com/equation?tex=w^t_{i,j}" alt="w^t_{i,j}" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= 是 i.i.d，那么 " alt=" 是 i.i.d，那么 " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= E(w^t_{i,j})=0, Var(w^t_{i,j})=r_t" alt=" E(w^t_{i,j})=0, Var(w^t_{i,j})=r_t" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= ；2. " alt=" ；2. " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=h^{t-1}_i" alt="h^{t-1}_i" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= 独立于 " alt=" 独立于 " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=w^t_{i,j}" alt="w^t_{i,j}" class="ee_img tr_noresize" eeimg="1"> $ (该层每个元素的输入，与其系数独立)

不考虑激活函数，那么 $ <img src="https://www.zhihu.com/equation?tex={\bf h}^t={\bf W}^t{\bf h}^{t-1}" alt="{\bf h}^t={\bf W}^t{\bf h}^{t-1}" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= ，这里的 " alt=" ，这里的 " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex={\bf W}^t \in \mathbb R^{n_t \times n_{t-1}}" alt="{\bf W}^t \in \mathbb R^{n_t \times n_{t-1}}" class="ee_img tr_noresize" eeimg="1"> $ 

可得其**正向均值** $ <img src="https://www.zhihu.com/equation?tex=E(h_i^t)" alt="E(h_i^t)" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= = 0 ；**正向方差** " alt=" = 0 ；**正向方差** " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=Var(h_i^t)=n_{t-1}r_tVar(h_j^{t-1})" alt="Var(h_i^t)=n_{t-1}r_tVar(h_j^{t-1})" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= ,希望输入的方差和输出方差一样，那么 " alt=" ,希望输入的方差和输出方差一样，那么 " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=n_{t-1}r_t=1" alt="n_{t-1}r_t=1" class="ee_img tr_noresize" eeimg="1"> $ ，n~t~ 是第t层的神经元个数

同理**反向均值** $ <img src="https://www.zhihu.com/equation?tex=E\left(\frac {\part l}{\part h_i^{t-1}} \right)=0" alt="E\left(\frac {\part l}{\part h_i^{t-1}} \right)=0" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= ；**反向方差** " alt=" ；**反向方差** " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=Var\left(\frac {\part l}{\part h_i^{t-1}} \right)=n_tr_tVar\left(\frac {\part l}{\part h_j^{t}} \right)" alt="Var\left(\frac {\part l}{\part h_i^{t-1}} \right)=n_tr_tVar\left(\frac {\part l}{\part h_j^{t}} \right)" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= ，为使两方差相等， " alt=" ，为使两方差相等， " class="ee_img tr_noresize" eeimg="1"> $n_tr_t=1" alt="其中a，b都是常数

如何做到呢？

假设 1. 第t层的每一行每一列的权重 $ <img src="https://www.zhihu.com/equation?tex=w^t_{i,j}" alt="w^t_{i,j}" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= 是 i.i.d，那么 " alt=" 是 i.i.d，那么 " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= E(w^t_{i,j})=0, Var(w^t_{i,j})=r_t" alt=" E(w^t_{i,j})=0, Var(w^t_{i,j})=r_t" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= ；2. " alt=" ；2. " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=h^{t-1}_i" alt="h^{t-1}_i" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= 独立于 " alt=" 独立于 " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=w^t_{i,j}" alt="w^t_{i,j}" class="ee_img tr_noresize" eeimg="1"> $ (该层每个元素的输入，与其系数独立)

不考虑激活函数，那么 $ <img src="https://www.zhihu.com/equation?tex={\bf h}^t={\bf W}^t{\bf h}^{t-1}" alt="{\bf h}^t={\bf W}^t{\bf h}^{t-1}" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= ，这里的 " alt=" ，这里的 " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex={\bf W}^t \in \mathbb R^{n_t \times n_{t-1}}" alt="{\bf W}^t \in \mathbb R^{n_t \times n_{t-1}}" class="ee_img tr_noresize" eeimg="1"> $ 

可得其**正向均值** $ <img src="https://www.zhihu.com/equation?tex=E(h_i^t)" alt="E(h_i^t)" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= = 0 ；**正向方差** " alt=" = 0 ；**正向方差** " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=Var(h_i^t)=n_{t-1}r_tVar(h_j^{t-1})" alt="Var(h_i^t)=n_{t-1}r_tVar(h_j^{t-1})" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= ,希望输入的方差和输出方差一样，那么 " alt=" ,希望输入的方差和输出方差一样，那么 " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=n_{t-1}r_t=1" alt="n_{t-1}r_t=1" class="ee_img tr_noresize" eeimg="1"> $ ，n~t~ 是第t层的神经元个数

同理**反向均值** $ <img src="https://www.zhihu.com/equation?tex=E\left(\frac {\part l}{\part h_i^{t-1}} \right)=0" alt="E\left(\frac {\part l}{\part h_i^{t-1}} \right)=0" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= ；**反向方差** " alt=" ；**反向方差** " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=Var\left(\frac {\part l}{\part h_i^{t-1}} \right)=n_tr_tVar\left(\frac {\part l}{\part h_j^{t}} \right)" alt="Var\left(\frac {\part l}{\part h_i^{t-1}} \right)=n_tr_tVar\left(\frac {\part l}{\part h_j^{t}} \right)" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= ，为使两方差相等， " alt=" ，为使两方差相等， " class="ee_img tr_noresize" eeimg="1"> $n_tr_t=1" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=r_t(n_{t-1}+n_t)/2=1 \to r_t = 2/(n_{t-1}+n_t)" alt="r_t(n_{t-1}+n_t)/2=1 \to r_t = 2/(n_{t-1}+n_t)" class="ee_img tr_noresize" eeimg="1">

对于正态分布

<img src="https://www.zhihu.com/equation?tex=N\left(0, \sqrt{2/(n_{t-1}+n_t)}\,\right)
" alt="N\left(0, \sqrt{2/(n_{t-1}+n_t)}\,\right)
" class="ee_img tr_noresize" eeimg="1">
对于均匀分布

<img src="https://www.zhihu.com/equation?tex=U\left(-\sqrt{6/(n_{t-1}+n_t)}, \sqrt{6/(n_{t-1}+n_t)}  \,\right)
" alt="U\left(-\sqrt{6/(n_{t-1}+n_t)}, \sqrt{6/(n_{t-1}+n_t)}  \,\right)
" class="ee_img tr_noresize" eeimg="1">


**激活函数 σ**

同样根据正向反向的方差分析，我们希望激活函数类似于 `f(x)=x`

检查常用激活函数，使用泰勒展开，看其函数值是否在零点附近接近x

<img src="https://www.zhihu.com/equation?tex=\begin{align}

sigmoid(x)&=\frac 1 2 + \frac x 4 - \frac{x^3} {48} + o(x^5)\\
tanh(x) &= 0+x-\frac {x^3} 3 +o(x^5)\\
relu(x) &= 0 + x \quad for\,x\ge 0

\end{align}
" alt="\begin{align}

sigmoid(x)&=\frac 1 2 + \frac x 4 - \frac{x^3} {48} + o(x^5)\\
tanh(x) &= 0+x-\frac {x^3} 3 +o(x^5)\\
relu(x) &= 0 + x \quad for\,x\ge 0

\end{align}
" class="ee_img tr_noresize" eeimg="1">
可以看见tanh和relu都还不错，sigmoid有点8行，但可以对其进行调整使得

<img src="https://www.zhihu.com/equation?tex=scaled \,sigmoid = 4\times sigmoid(x)-2
" alt="scaled \,sigmoid = 4\times sigmoid(x)-2
" class="ee_img tr_noresize" eeimg="1">
几个函数对比如下

```python
import torch
import matplotlib.pyplot as plt
from d2l import torch as d2l



x = torch.arange(-2.5, 2.5, 0.1)
ssigmoid = 4 * torch.sigmoid(x) - 2
sigmoid = torch.sigmoid(x)
tanh = torch.tanh(x)
relu = torch.relu(x)

fig, ax = plt.subplots(figsize=(5, 2.7))
ax.plot(x, ssigmoid, 'b-',label='scaled sigmoid')
ax.plot(x, sigmoid, 'g-.',label='sigmoid')
ax.plot(x, tanh,'r--', label='tanh')
ax.plot(x, relu,'y:', label='relu')
ax.legend()
```

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211217190031300.png" alt="image-20211217190031300" style="zoom: 50%;" />



## QA环节

Q：关于机器学习和数学？

A：深度学习相对来说不太需要数学。但数学很重要，就像计算机的内存大小一样（决定有多复杂），而代码能力就好比CPU频率（决定有多快）



Q：为什么要用正态分布假设？

A：一个是简单，一个是大数定律



Q：强制是每一层的输出特征均值为0，方差为定值，会不会影响网络的表达能力？

A：不会，数值范围只是给定一个区间（相当于整体压缩在一个区间内），故不会影响其表达性



# ==Kaggle房价预测实战==

## 下载和缓存数据集

这里插播一条关于**函数的可变参数**的定义

+ `*args`：发送一个非键值对的可变数量的参数列表给函数，读作 `list of augments`
+ `**kwargs`：发送一个键值对的可变数量的参数列表给函数
+ 如果想要在函数内使用带有名称的变量（像字典那样），那么使用`**kwargs`
+ 定义可变参数的目的是为了简化调用。`*`和`**`在此处的作用：打包参数

使用时，后面的名字可以更改，只是约定俗成而已。比如可以用`*vals`代表非键值对的可变数量的参数，`**parms`代表可变数量的键值对参数。当要同时使用`*args`和`**kwargs`时，`*args`必须写在`**kwargs`之前
示例代码如下：

```python
def test_args(*args):
    print(args)
 
 
def test_kwargs(**kwargs):
    print(kwargs)
    print(type(kwargs))
    for key, value in kwargs.items():
        print("{} == {}".format(key, value))
 
 
def test_all(*args,**kwargs):
    print(args)
    print(kwargs)

test_args('nep', 'blanc', 'noire')
print('----------')
test_kwargs(puple='nep', white='blanc', black='noire')
print('----------')
test_all('nep', 'blanc', puple='nep', white='blanc')

# ('nep', 'blanc', 'noire')
# ----------
# {'puple': 'nep', 'white': 'blanc', 'black': 'noire'}
# <class 'dict'>
# puple == nep
# white == blanc
# black == noire
# ----------
# ('nep', 'blanc')
# {'puple': 'nep', 'white': 'blanc'}
```



再插播一下**sha1加密**

**SHA1算法简介**

安全哈希算法（Secure Hash Algorithm）主要适用于数字签名标准（Digital Signature Standard DSS）里面定义的数字签名算法（Digital Signature Algorithm DSA）。对于长度小于2^64位的消息，SHA1会产生一个160位的消息摘要。当接收到消息的时候，这个消息摘要可以用来验证数据的完整性。在传输的过程中，数据很可能会发生变化，那么这时候就会产生不同的消息摘要。

SHA1有如下特性：不可以从消息摘要中复原信息；两个不同的消息不会产生同样的消息摘要。



==正式开始！==

在这⾥，我们实现⼏个函数来⽅便下载数据。⾸先，我们维护字典DATA_HUB，其将数据集名的字符串映射到数据集相关的⼆元组上，这个⼆元组包含数据集的url和验证⽂件完整性的sha-1密钥。所有这样的数据集都托管在地址为DATA_URL的站点上
数据集名为k， 二元组（url和sha-1秘=密钥）为v

```python
import hashlib
import os
import tarfile
import zipfile
import requests

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
```



下⾯**定义download函数⽤来下载数据集**，将数据集缓存在本地⽬录（默认情况下为../data中，并返回下载⽂件的名称。如果缓存⽬录中已经存在此数据集⽂件，并且其sha-1与存储在DATA_HUB中的相匹配，就使⽤缓存的⽂件，以避免重复的下载

```python
def download(name, cache_dir = os.path.join('..','data')):
    '''下载一个DATA_HUB中的文件，返回本地文件名'''
    # assert后面是不满足条件时显示的字符串，如果想执行操作就try except
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}."
    # 根据传入的name获取url和密钥
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    # 将url按 '/' 切五花肉 获取五花肉块列表 返回最后一块肉
    fname = os.path.join(cache_dir, url.split('/')[-1])

    if os.path.exists(fname):
        # 创建一个sha1加密的对象
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576) # 2的20次方bytes（字节） 即1MB
                if not data:
                    break  # 不用再判断密钥了直接给我去下载吧
                sha1.update(data)
        # 如果生成的密钥和之前DATA_HUB中保存的秘钥一样那么就直接返回其文件名，不用下载
        if sha1.hexdigest() == sha1_hash:
            print('涅普！文件已存在！')
            return fname
    
    # 那就只有下载了噻
    print(f'Nep！正在从{url}中下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
```

再定义两个实用函数：一个是**下载并解压一个zip或tar文件**，另一个是**将本书中使用的所有数据集从DATA_HUB下载到缓存目录中**

```python
def download_extract(name, folder=None):
    '''下载并解压zip/tar文件'''
    fname = download(name)
    base_dir = os.path.dirname(fname)  # 去掉文件名，返回目录base_dir
    data_dir, ext = os.path.splitext(fname)  # 获取文件名和后缀名
    if ext == '.zip':
        fp = zipfile.ZipFile(fname,'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname,'r')
    else:
        assert False, 'Nep! 只可以解压zip/tar哦！'
    fp.extractall(base_dir)    # 解压到当前目录base_dir
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    '''下载DATA_HUB中的所有文件'''
    for name in DATA_HUB:
        download(name)
```



## 访问并读取数据集

**下载数据，并使用pandas读入处理**

```python
%matplotlib inline
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
```

方便起见，使用上面定义的本下载并缓存Kaggle房屋数据集

```python
DATA_HUB['kaggle_house_train'] = (
# 二元组分别对应其url和密钥
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
```

```python
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
```

![image-20211220220103706](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211220220103706.png)

查看一下数据

```python
train_data.shape, test_data.shape

# ((1460, 81), (1459, 80))
```

```python
# 查看前四个样本的前四个后最后两个特征，以及相应的标价
train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]
```

![image-20211220220234473](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211220220234473.png)



## 数据预处理

我们可以看到，在每个样本中，第⼀个特征是ID，这有助于模型识别每个训练样本。虽然这很⽅便，但它不携带任何⽤于预测的信息。因此，**在将数据提供给模型之前，我们将其id信息从数据集中删除**

```python
# 去掉ID后的所有样本,都表示对于训练集/测试集的所有样本，从下标1一直到最后，使用pd的concat将二者合并
all_features = pd.concat((train_data.iloc[:,1:-1], test_data.iloc[:, 1:]))
all_features

# 2919 rows × 79 columns
```

 

对于==数值特征==，将所有**缺失值替换**为相应特征的平均值。然后进行**feature scaling**（先标准化再替换会更方便，因为标准化后均值就是0了）

```python
# 假设不是'object'类的数据就是数值类，把所有数值特征提取出来
numeric_feaures = all_features.dtypes[all_features.dtypes != 'object'].index
# 对数值特征进行标准化(实际情况拿不到测试集的时候可以用训练集的均值方差代替)
all_features[numeric_feaures] = all_features[numeric_feaures].apply(
    lambda x: (x-x.mean()) / (x.std()))
# 把NAN变成0(也就是变成均值，妙啊！)
all_features[numeric_feaures] = all_features[numeric_feaures].fillna(0)
```



对于==离散值特征==，我们使用**one_hot独热编码**来替代

```python
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape

# (2919, 331)
```

对比之前79个特征，现在直接331个特征了（离散值被做成了one_hot）！

<img src="https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211220225156441.png" alt="image-20211220225156441" style="zoom:80%;" />



将训练集/测试集的特征和训练集的特征标签 **从`pandas`格式中提取Numpy格式，并转化为tensor**

```python
n_train = train_data.shape[0] # 训练集样本数
# 由于numpy默认为64位浮点，故转为32位
# 提取训练集和测试集的特征为tensor
train_features = torch.tensor(all_features[:n_train].values,
                             dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values,
                             dtype=torch.float32)
# 提取训练集的标签为tensor
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1,1),
                           dtype=torch.float32)
```



## 训练函数

⾸先，我们训练⼀个带有损失平⽅的线性模型。毫不奇怪，我们的线性模型不会让我们在竞赛中获胜，但线性模型提供了⼀种健全性检查，以查看数据中是否存在有意义的信息。如果我们在这⾥不能做得⽐随机猜测更好，那么我们很可能存在数据处理错误。如果⼀切顺利，线性模型将作为基线模型，让我们直观地知道简单的模型离报告最好的模型有多近，让我们感觉到我们应该从更酷炫的模型中获得多少收益

```python
loss = nn.MSELoss()
in_features = train_features.shape[-1] # 提取所有训练特征

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net
```




<img src="https://www.zhihu.com/equation?tex=\frac {y-\hat{y}} {y}$ <img src="https://www.zhihu.com/equation?tex= ，而不是绝对误差 " alt=" ，而不是绝对误差 " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=y-\hat{y}" alt="y-\hat{y}" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= （北京和旺苍的房价预测误差10w的权重不一样），也就是 " alt=" （北京和旺苍的房价预测误差10w的权重不一样），也就是 " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=\frac {\hat{y}}{y}" alt="\frac {\hat{y}}{y}" class="ee_img tr_noresize" eeimg="1"> $ 越接近1越好。解决这种问题的方法一般是取 log （将除法进行转化）

用 δ 来衡量误差，即对于（y_hat 比 y 大或小了都不好）
" alt="\frac {y-\hat{y}} {y}$ <img src="https://www.zhihu.com/equation?tex= ，而不是绝对误差 " alt=" ，而不是绝对误差 " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=y-\hat{y}" alt="y-\hat{y}" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex= （北京和旺苍的房价预测误差10w的权重不一样），也就是 " alt=" （北京和旺苍的房价预测误差10w的权重不一样），也就是 " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=\frac {\hat{y}}{y}" alt="\frac {\hat{y}}{y}" class="ee_img tr_noresize" eeimg="1"> $ 越接近1越好。解决这种问题的方法一般是取 log （将除法进行转化）

用 δ 来衡量误差，即对于（y_hat 比 y 大或小了都不好）
" class="ee_img tr_noresize" eeimg="1">
e^{-\delta} \le \frac{\hat{y}}{y} \le e^\delta

<img src="https://www.zhihu.com/equation?tex=我们希望 δ 越接近 0 越好，取对数即
" alt="我们希望 δ 越接近 0 越好，取对数即
" class="ee_img tr_noresize" eeimg="1">
|\log y-\log \hat y|\le\delta

<img src="https://www.zhihu.com/equation?tex=故定义以下**对数均方根误差**
" alt="故定义以下**对数均方根误差**
" class="ee_img tr_noresize" eeimg="1">
\sqrt{\frac1n \sum_{i=1}^n(\log y_i-\log \hat{y}_i)^2}
$$

```python
def log_rmse(net, features, labels):
    # 对于一个net的输出如果是inf的话就将其变为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()  # 取出单元素张量的元素值并返回该值，保持原元素类型不变
```



**训练函数借助Adam优化器（RMSprop + Momentum），使用weight decay**

```python
def train(net, train_features, train_labels, test_features, test_labels,
         num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []    # 显示在图像上一个epoch更新一次的loss
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)   # 为什么训练的时候不用对数均方误差？而只在评估时使用？
            l.backward()
            optimizer.step()
        # 扫完一个epoch后记录其log_rmse
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            # 计算测试集的误差
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```



## 实现K折交叉验证

对于第i折，返回其训练集和测试集

```python
def get_k_fold_data(k, i, X, y):
    assert k > 1  # k肯定大于1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        # 切片，nep[slice(2,4)] == nep[2:4]
        idx = slice(j * fold_size, (j + 1) * fold_size)
        # 取出当前折
        X_part, y_part = X[idx,:], y[idx]
        if j == i:    # vaildation set
            X_vaild, y_vaild = X_part, y_part
        elif X_train is None:
            # 为空的话就先赋值
            X_train, y_train = X_part, y_part
        else:
            # 非空的话就concat
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_vaild, y_vaild
```

返回k折交叉验证后的平均训练误差和验证误差

```python
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls_sum, vaild_ls_sum = 0, 0
    # 进行k折交叉验证
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        # *data 表示打包非键值对参数data，对应tupple (训练特征，训练标签， 测试特征， 测试标签)
        # trian()返回的是tupple (训练误差， 测试误差) 可以直接像这样用两个元素去接
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_ls_sum += train_ls[-1]   # -1表示取这一轮最后一个epoch的loss
        vaild_ls_sum += valid_ls[-1]
        if i == 0:
            # 自变量 因变量 标签  x轴 图例说明  y轴
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                    xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                    legend=['train', 'valid'], yscale='log')
        print(f'Nep！fold{i+1},train log rmse {float(train_ls[-1]):f}, '
             f'valid log rmse {float(valid_ls[-1]):f}')
    return train_ls_sum / k, vaild_ls_sum / k
```



## 模型选择

这里我们先选择了⼀组未调优的超参数，然后自己来改进模型。找到⼀个好的选择可能需要时间， 这取决于⼀个⼈优化了多少变量。有了⾜够⼤的数据集和合理设置的超参数，K折交叉验证往往对多次测试 具有相当的适应性。然而，如果我们尝试了不合理的⼤量选项，我们可能会发现验证效果不再代表真正的误差

```python
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)

print('k折交叉验证全部搞定！')
print(f'{k}折验证：平均训练log rmse：{float(train_l):f},'
     f'平均验证log rmse：{float(valid_l):f}')
```

![image-20211221230033007](https://raw.githubusercontent.com/NepNeppp/Markdown4Zhihu/master/Data/1_神经网络的基本概念和操作/image-20211221230033007.png)



**对于较大的数据集以及较复杂的模型，我们可以先使用一小批数据对超参数进行调试得到一个大致范围，再使用大的数据集进行调整**



# ==PyTorch神经网络基础==

## 模型构造

随着时间的推移，深度学习库已经演变成提供越来越粗糙的抽象。就像半导体设计师从指定晶体管到辑电路再到编写代码⼀样，神经⽹络研究⼈员已经从考虑单个⼈⼯神经元的⾏为转变为从**层**的⻆度构思⽹络，现在通常在设计结构时考虑的是更粗糙的**块**（block）。**当需要更⼤的灵活性时，我们需要定义⾃⼰的块**。



先回忆一下多层感知机的代码

```python
import torch
from torch import nn
from torch.nn import functional as F   # 实现了大量的常用函数

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

# 生成两个样本
X = torch.rand(2, 20)
net(X)

# tensor([[-0.0540, -0.1976, -0.3475,  0.2234,  0.0020,  0.2708,  0.1132,  0.1311,
#          -0.0524,  0.1620],
#         [-0.0737, -0.2125, -0.2790,  0.1669, -0.2071,  0.3636,  0.0843,  0.1188,
#          -0.1503,  0.1745]], grad_fn=<AddmmBackward>)
```



自定义块

```python
# 任何一个层或块都是Module的子类，自定义的话需要继承Moudle和重写__init__()和forward()函数
class MLP(nn.Module):
    # 初始化神经网络结构
    def __init__(self):
        super().__init__()   # 调用父类Module的初始化函数
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
    
    # 定义向前传播的激活函数（反向传播不需要你管）
    # 先把输入放到hidden层里面，再调用nn.Module.functional中的relu()
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
```

```python
net = MLP()
net(X)

# tensor([[ 0.1181,  0.1892,  0.1640,  0.0123,  0.0431, -0.0658,  0.1960,  0.0392,
#           0.0977,  0.1609],
#         [ 0.0467,  0.1841,  0.0023, -0.0735,  0.0148, -0.0629,  0.1294, -0.0686,
#           0.0621,  0.0792]], grad_fn=<AddmmBackward>)
```



顺序块（就是让你理解一下`nn.Sequential()`在干啥）

```python
class MySequential(nn.Module):
    # *args读作'list of input arguments'，里面是一个块的序列
    def __init__(self, *args):
        super().__init__()
        for block in args:
            # _modules是pytorch特殊的一个容器，放在里面的都是一些层，是一个有序字典（OrderedDict）
            self._modules[block] = block
    
    # 一块一块地往前传
    def forward(self, X):
        # 这不就是dict的values吗？按顺序一层一层往前调用
        for block in self._modules.values():
            X = block(X)
        return X

net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```



在正向传播函数中执⾏代码==**（誰よりも自由で）**==

*好比做一个hamburger，你可以凭自己喜好放各种肉饼、芝士、蔬菜*（layer，active function， freestyle）

```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Nep希望有一个随机的weight，但是不参加训练
        self.rand_weight_nep = torch.rand((20, 20), requires_grad = False)
        self.linear_nep = nn.Linear(20, 20)
    
    # Nep可以自己乱整，想干啥干啥
    def forward(self, X):
        # 乱整中，无实义，意思是以后你可以对层之间的操作自定义
        X = self.linear_nep(X)
        X = F.relu(torch.mm(X, self.rand_weight_nep) + 1)
        X = self.linear_nep(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

net = FixedHiddenMLP()
net(X)

# tensor(-0.1687, grad_fn=<SumBackward0>)
```



混合搭配各种组合块

自己随意搭配一个~~雀巢~~（嵌套）网络

```python
class NestMLP(nn.Module):
# Sequential和linear的嵌套
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)
        
    def forward(self, X):
        return self.linear(self.net(X))

# 再来，Sequential也可以继续嵌套
chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)

# tensor(0.2182, grad_fn=<SumBackward0>)
```



## 参数管理

例子：对于单隐藏层的MLP

```python
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2,4))
net(X)

# tensor([[0.7092],
#         [0.6509]], grad_fn=<AddmmBackward>)
```



### 参数的访问

访问参数状态， 返回一个**OrderedDict**

```python
# 整个网络的参数
net.state_dict()

# OrderedDict([('0.weight',
#               tensor([[ 2.7738e-01,  3.7378e-01,  2.0394e-01, -3.2199e-02],
#                       [ 4.8738e-01,  6.7645e-04, -2.5393e-01,  4.4538e-01],
#                       [-1.4785e-01,  3.7953e-01, -3.6851e-01, -1.8559e-01],
#                       [ 2.7460e-01, -4.5386e-01, -3.9325e-01, -1.6626e-01],
#                       [-8.7663e-02, -3.5089e-01,  1.5684e-01, -2.1210e-01],
#                       [ 2.8558e-01,  2.5885e-01,  4.4810e-01, -3.3963e-01],
#                       [ 4.0101e-01,  4.5958e-01, -2.7281e-01, -3.4963e-02],
#                       [ 1.1469e-01, -3.9816e-04, -2.7695e-02,  2.6692e-01]])),
#              ('0.bias',
#               tensor([ 0.4660, -0.1200, -0.2365, -0.2294,  0.4426,  0.3392,  0.1555, -0.4485])),
#              ('2.weight',
#               tensor([[ 0.3448, -0.3285, -0.0938,  0.2211,  0.0133,  0.2955,  0.0905,  0.1235]])),
#              ('2.bias', tensor([0.2061]))])
```

```python
# 指定层的参数
# net可以看做是一个序列，故net[2]表示之前的第三层nn.Linear(8, 1)；参数可以理解为一种状态（因为可变）故叫state_dict()
print(net[2].state_dict())

# OrderedDict([('weight', tensor([[ 0.3448, -0.3285, -0.0938,  0.2211,  0.0133,  0.2955,  0.0905,  0.1235]])), ('bias', tensor([0.2061]))]
```

根据状态的顺序字典的 Key 访问参数，具体到 `weight` 和 `bias`，返回**tensor**

```python
net.state_dict()['2.weight'].data

# tensor([[ 0.3448, -0.3285, -0.0938,  0.2211,  0.0133,  0.2955,  0.0905,  0.1235]])
```

类似地，具体访问目标参数，返回

```python
# 访问目标参数
print(type(net[2].bias))
print(net[2].bias)
print(net[2].weight.data)
print(net[2].bias.data)
print(net[2].bias.grad)

# OrderedDict([('weight', tensor([[ 0.3448, -0.3285, -0.0938,  0.2211,  0.0133,  0.2955,  0.0905,  0.1235]])), ('bias', tensor([0.2061]))])
# <class 'torch.nn.parameter.Parameter'>
# Parameter containing:
# tensor([0.2061], requires_grad=True)
# tensor([[ 0.3448, -0.3285, -0.0938,  0.2211,  0.0133,  0.2955,  0.0905,  0.1235]])
# tensor([0.2061])
# None
```



一次性访问所有参数

```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

# ('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
# ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
```



从嵌套块搜集参数，先定义嵌套块

```python
# 块⼯⼚
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套4次 block1
        net.add_module(f'block{i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)

# tensor([[-0.4434],
#         [-0.4434]], grad_fn=<AddmmBackward>)
```

查看一下rgnet的结构

```python
rgnet

# Sequential(
#   (0): Sequential(
#     (block0): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#     (block1): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#     (block2): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#     (block3): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#   )
#   (1): Linear(in_features=4, out_features=1, bias=True)
# )

# 整体是一个Sequential，然后又分为 Sequential[0] 和 Sequential[1]，前者对应刚刚的4块合并， 后者对面后面添加的 Linear(4, 1)
```

```python
# 访问第一个 Sequential 的 block1
rgnet[0][1]

#  Sequential(
#    (0): Linear(in_features=4, out_features=8, bias=True)
#    (1): ReLU()
#    (2): Linear(in_features=8, out_features=4, bias=True)
#    (3): ReLU()
#  ))
```

```python
# 访问 第一个Sequential的 block1的 第1层的 bias参数
rgnet[0][1][0].bias.data

# tensor([-0.0300, -0.3123,  0.3716,  0.1834, -0.3541, -0.0904,  0.1097, -0.0884])
```



### 参数的初始化

深度学习框架提供默认随机初始化。然而，我们经常希望根据其他规则初始化权重。深度学习框架提供了最常⽤的规则，也允许创建⾃定义初始化⽅法。



内置的初始化

```python
# 对传入module参数进行初始化
def init_normal(m):
    # 如果是全连接层我们再做初始化，不然不管
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

# apply()，可以理解为对net里面所有的module进行遍历
net.apply(init_normal)
net[0].weight.data[0]

# tensor([ 0.0010, -0.0074,  0.0259,  0.0027])
```

初始成常数（乱整），从API的角度可以做但是算法的角度万万不可（恩达说过，否则每一层的元素梯度下降都完全一样，即最后同一个layer的感知机参数都一模一样）

```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]

# (tensor([1., 1., 1., 1.]), tensor(0.))
```



对不同层用不同的初始化方法

例如，下⾯我们使⽤Xavier初始化⽅法初始化第⼀层，然后第⼆层初始化为常量值4143283（玩）

```python
def xavier(m):
    if type(m) == nn.Linear:
        # nn.init.xavier_normal_() and nn.init.xavier_uniform_() 分别对应之前讲的Xavier正态和均匀
        nn.init.xavier_uniform_(m.weight)
def init_4143283(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 4143283)

net[0].apply(xavier)
net[2].apply(init_4143283)
print(net[0].weight.data[0])
print(net[2].weight.data)

# tensor([0.2825, 0.0268, 0.6628, 0.4849])
# tensor([4143283., 4143283., 4143283., 4143283., 4143283., 4143283., 4143283.,
#         4143283.])
```



自定义初始化（不常用，玩一玩）

```python
def my_init(m):
    if type(m) == nn.Linear:
        print(
            'Init',
            *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5  # 保留绝对值大于等于5的参数，否则置零

net.apply(my_init)
net[0].weight[:2]

# Init weight torch.Size([8, 4])
# Init weight torch.Size([1, 8])
# tensor([[-0.0000,  5.5843,  8.6725,  6.5915],
#         [-0.0000, -7.9470,  7.1977, -9.0225]], grad_fn=<SliceBackward>)
```



直接访问参数并进行修改

```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]

# tensor([42.0000,  5.6274,  6.3860,  6.2191])
```



### 参数绑定

有时我们希望在**多个层间共享参数**。让我们看看如何优雅地做这件事。在下⾯，我们定义⼀个稠密层，然后 使⽤它的参数来设置另⼀个层的参数。

```python
# 让不同层share相同的权重
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8, 1))

net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])

# 改变参数后其他shared会一起改变
net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])

# tensor([True, True, True, True, True, True, True, True])
# tensor([True, True, True, True, True, True, True, True])
```



## 自定义层

跟自定义网络没什么区别

深度学习成功背后的⼀个因素是，可以⽤创造性的⽅式组合⼴泛的层，从而设计出适⽤于各种任务的结构。例如，研究⼈员发明了专⻔⽤于处理图像、⽂本、序列数据和执⾏动态编程的层。早晚有⼀天，你会遇到或要⾃⼰发明⼀个在深度学习框架中还不存在的层。在这些情况下，你必须构建⾃定义层。



### 不带参数的层

下⾯的CenteredLayer类要从其输⼊中减去均值（wow不会BN就要这么搞吧~）。要构建它，我们只需继承基础层类并实现正向传播功能

```python
import torch
import torch.nn.functional as F
from torch import nn

class CenteredLayer(nn.Module):
    # python3 可以不写，会默认加上
#   def __init__(self):
#       super().__init__()
        
    def forward(self, X):
        return X - X.mean()
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))、

# tensor([-2., -1.,  0.,  1.,  2.])
```

将层作为组件加到模型中，~~涂层芝士~~

```python
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

Y = net(torch.rand(4, 8))
Y.mean()

# tensor(-2.3283e-09, grad_fn=<MeanBackward0>)
```



### 带参数的层

现在，让我们实现**⾃定义版本的全连接层**。回想⼀下，该层需要两个参数，⼀个⽤于表⽰**权重**，另⼀个⽤于表⽰**偏置项**。在此实现中，我们使⽤ReLU作为**激活函数**。该层需要输⼊参数：in_units和units，分别表⽰输⼊和输出的数量。

通过 nn.Parameter() 自定义参数层

```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        # [-1, 1]之间的均匀分布
        # 所有的加梯度加名字都在parammeter中完成
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        
    def forward(seld, X):
        linear = torch.matmul(X, self.weight.data) + self,bias.data
        return F.relu(linear)
    
dense = MyLinear(5, 3)
dense.weight

# Parameter containing:
# tensor([[ 0.1729,  0.9538,  0.8921],
#         [ 0.7559, -0.8837, -0.0122],
#         [ 0.1766, -1.0734,  0.1221],
#         [ 0.4778,  0.2188,  1.3684],
#         [-0.5119,  0.8046,  0.4938]], requires_grad=True)
```



## 读写文件

当运⾏⼀个耗时较⻓的训练过程时，最佳的做法是定期保存中间结果（检查点），以确保在服务器电源被 不小⼼断掉时不会损失⼏天的计算结果。因此，现在是时候学习如何加载和存储权重向量和整个模型



**首先我们要了解pytorch有关保存和加载模型的三个核心函数**

1. torch.save: 该函数用python的pickle实现序列化，并将序列化后的object放到硬盘。
2. torch.load: 用pickle将object从硬盘中反序列化到内存中。
3. torch.nn.Module.load_state_dict: 通过反序列化后的state_dict 来读取模型的训练参数

==**state_dict**==
state_dict是一个简单的python字典，它映射模型的每一层到每一层的参数tensor. 这里值得注意的是，只有有可学习参数的层（例如convolutional layers，linear layers等等）以及已经注册的buffers才会在state_dict中存在。torch.optim对象同样也有state_dict,它处指出优化器的状态和优化器使用的超参数，这用于你需要再次训练的模型上。



存储和读取张量

```python
import torch
from torch import nn
from torch import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load("x-file")
x2

# tensor([0, 1, 2, 3])
```

存储张量列表list

```python
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)

# (tensor([0, 1, 2, 3]), tensor([0., 0., 0., 0.]))
```

存储或读取从字符串映射到张量的字典dict

```python
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2

# {'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}
```



存储和读取模型参数，（相当于其他框架，pytorch不方便将整个模型存储下来，实在要存可以用torchscript），故我们只存储其权重

还是先定义一个MLP

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))
    
net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

将模型的state_dict拿出来并保存

```python
torch.save(net.state_dict(), 'mlp.params')
```

读取存储的参数，实现原始MLP的一个备份

```python
clone = MLP()    # 复制其结构, 此时为随机化参数
clone.load_state_dict(torch.load('mlp.params'))   # 读取参数并overwrite之前的参数
clone.eval()

# MLP(
#   (hidden): Linear(in_features=20, out_features=256, bias=True)
#   (output): Linear(in_features=256, out_features=10, bias=True)
# )
```

比较一下对于同样的输入，两个MLP的输出

```python
Y_clone = clone(X)
Y_clone == Y

# tensor([[True, True, True, True, True, True, True, True, True, True],
#         [True, True, True, True, True, True, True, True, True, True]])
```



## QA环节

Q：在将字符串变量转换成one_hot编码时内存炸掉怎么办？

A：one_hot根据每一个类别设置一个编码，如果类别太多内存就会爆炸。可以使用**稀疏矩阵**进行存储；或者考虑**不使用one_hot**进行离散化（例如房屋预测中，房屋的地址、summary等就不适合使用one_hot），使用NLP的backforwards，之后会讲



Q：自定义MLP时，我们只实例化了`__init__()`和`forward()`函数，为什么可以直接调用`net(X)`而不是`net.forward(X)`呢？

A：`net(X)`实际上调用的是`net.__call__()`,由于继承了`nn.Moudle`，它将`__call__()`和`forward()`进行了等价，故二者均可



Q：forward函数里的函数是怎么得出来的？

A：根据网络而定，对着paper里面的公示敲出来的（doge）



Q：kaiming初始化？

A：何凯明提出的一种初始化方法，类似于Xavier。不要太迷信参数初始化，它的目的主要是让模型一开始的时候不要炸掉，各个参数在同一个尺度上，相对来说不会太影响后面的结果



Q：遇到不可导点怎么办？

A：在实际的数值计算中很难遇到不可导点，一般来说都是离散的点，就算遇到了可以去左、右导数或者其他处理。实际数值计算中不会出现类似狄利克雷函数的情况。