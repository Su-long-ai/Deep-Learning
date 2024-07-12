import torch

x = torch.arange(4, dtype=torch.float32, requires_grad=True)  # 存储梯度
print(x.grad)

y = 2 * torch.dot(x, x)
print(y)

y.backward()  # 反向传播
print(x.grad)
print(x.grad == 4 * x)

x.grad.zero_()  # 梯度清零
y = x.sum()
y.backward()
print(x.grad)

x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)
