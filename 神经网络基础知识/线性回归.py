import torch
from torch.utils import data
from torch import nn
from d2l import torch as d2l


def load_array(data_arrays, batch_size_, is_train=True):  # PyTorch数据迭代器
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size_, shuffle=is_train)


net = nn.Sequential(nn.Linear(2, 1))  # 全连接层

net[0].weight.data.normal_(0, 0.01)  # 初始化模型参数
net[0].bias.data.fill_(0)

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))

loss = nn.MSELoss()  # 损失函数

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for x, y in data_iter:
        L = loss(net(x), y)
        trainer.zero_grad()
        L.backward()
        trainer.step()
    L = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {L:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
