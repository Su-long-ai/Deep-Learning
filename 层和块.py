import torch
from torch import nn
from torch.nn import functional as f


class MLP(nn.Module):  # 自定义块
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化
        # 这样，在类实例化时也可以指定其他函数参数
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, x):
        # ReLU的函数版本，其在nn.functional模块中定义
        return self.out(f.relu(self.hidden(x)))


class MySequential(nn.Module):  # 顺序块
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, x):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            x = block(x)
        return x
    
    
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数，因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # 使用创建的常量参数以及relu和mm函数
        x = f.relu(torch.mm(x, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        x = self.linear(x)
        # 将x的总和逐步缩小（除于2），直到它的总和的绝对值之和小于等于1，然后返回它的总和
        while x.abs().sum() > 1:
            x /= 2
        return x.sum()
    
    
net = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

X = torch.rand(2, 20)
print(net(X))

net = MLP()
print(net(X))

net = MySequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print(net(X))

net = FixedHiddenMLP()
print(net(X))

# 当然，我们可以按需求混合搭配各种组合块
