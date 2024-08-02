import torch
from torch import nn


def try_gpu(i=0):
    # 如果存在，则返回gpu(i)，否则返回cpu()
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    # 返回所有可用的gpu，如果没有gpu，则返回[cpu(),]
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


print(torch.cuda.device_count())  # 查询可用gpu的数量
print(try_gpu(), try_gpu(1), try_all_gpus())  # 只有1个gpu，集显不算

X = torch.ones(2, 3, device=try_gpu())  # 张量创建在gpu上，try_gpu()可以填数字指定第几个gpu
print(X)

Y = X + X
print(Y)

net = nn.Sequential(
    nn.Linear(3, 1)
)
net = net.to(device=try_gpu())
print(net(X))
print(net[0].weight.data.device)  # 确认模型参数存储在同一个GPU上
