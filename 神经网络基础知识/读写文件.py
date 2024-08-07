import torch
from torch import nn
# noinspection PyPep8Naming
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x_):
        return self.output(F.relu(self.hidden(x_)))


x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load('x-file')
print(x2)

y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
print((x2, y2))

mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), 'mlp.params')

# 创建一个新的模型实例
clone = MLP()
# 从文件加载保存的模型参数到克隆模型
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())  # 评估模式

Y_clone = clone(X)
print(Y_clone == Y)
