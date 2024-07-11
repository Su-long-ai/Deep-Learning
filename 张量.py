import torch

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())

x = x.reshape(3, 4)
print(x)

x = torch.zeros((2, 3, 4))
print(x)
x = torch.ones((2, 3, 4))
print(x)
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x)

# 可按元素运算
x = torch.exp(x)
print(x)

y = torch.arange(9, dtype=torch.float32).reshape(3, 3)
z = torch.cat((x, y), dim=0)  # 行拼接
print(z)
z = torch.cat((x, y), dim=1)  # 列拼接
print(z)

z = x == y
print(z)

z = x.sum()
print(z)

# 广播机制
x = torch.arange(3).reshape(3, 1)
y = torch.arange(2).reshape(1, 2)
print(x + y)

print(x[-1, :])
print(x[1:2, :])

x[2, 0] = 9
print(x)
x[0:2, :] = 3
print(x)

x = torch.arange(3).reshape(1, 3)
y = torch.arange(3).reshape(1, 3)
z = id(x)
x += y
print(id(x) == z)
x = x + y
print(id(x) == z)

x = x.numpy()
print(type(x))
x = torch.tensor(x)
print(type(x))
