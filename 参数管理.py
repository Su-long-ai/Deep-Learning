import torch
from torch import nn


def block1():
    return nn.Sequential(nn.Linear(4, 8),
                         nn.ReLU(),
                         nn.Linear(8, 4),
                         nn.ReLU()
                         )


def block2():
    net_ = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net_.add_module(f'block {i}', block1())
    return net_


def init_normal(m):  # 我们指定初始化方法
    if isinstance(m, nn.Linear):
        # 将所有权重参数初始化为标准差为0.01的高斯随机变量，并且将偏置参数设置为0
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


def init_constant(m):
    if isinstance(m, nn.Linear):
        # 将所有权重参数初始化为1，并且将偏置参数设置为0
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


def init_xavier(m):
    if isinstance(m, nn.Linear):
        # 使用Xavier均匀初始化
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 42)


def my_init(m):
    if isinstance(m, nn.Linear):
        params = [(name, param.shape) for name, param in m.named_parameters()]
        print("Init", *params[0])
        nn.init.uniform_(m.weight, -10, 10)
        # 将绝对值小于5的权重置为0
        m.weight.data *= m.weight.data.abs() >= 5


net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
X = torch.rand(size=(2, 4))
print(net(X))

# 参数访问
print(net[2].state_dict())  # 查看层的参数

print(type(net[2].bias))  # 查看偏置参数（第一种）
print(net[2].bias)
print(net[2].bias.data)

print(net[2].weight.grad is None)  # 此时梯度还未更新

# 遍历参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
print(net.state_dict()['2.bias'].data)  # 查看偏置参数（第二种）

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
print(rgnet)  # 查看层结构

# 1.rgnet = nn.Sequential(block2(), nn.Linear(4, 1))中的block2()
# 2.4次循环，从0开始，序号为1的循环
# 3.序号为1的循环里的第一层
print(rgnet[0][1][0].bias.data)

net.apply(init_normal)  # 应用初始化
print(net[0].weight.data[0], net[0].bias.data[0])

net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

net.apply(my_init)
print(net[0].weight[:2])

# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8),
                    nn.ReLU(),
                    shared,
                    nn.ReLU(),
                    shared,
                    nn.ReLU(),
                    nn.Linear(8, 1)
                    )
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
