import torch
import torch.nn.functional as f
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    # 这里ide让我用static，实际上并不是，就加了上面一句
    def forward(self, x):
        return x - x.mean()


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        # nn.Parameter确保权重和偏置在训练过程中被优化
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, x):
        linear_ = torch.matmul(x, self.weight) + self.bias
        return f.relu(linear_)


layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

Y = net(torch.rand(4, 8))
print(Y.mean())

linear = MyLinear(5, 3)
print(linear.weight)

print(linear(torch.rand(2, 5)))

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))
