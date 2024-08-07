import torch
from torch import nn
from d2l import torch as d2l
import os
import matplotlib.pyplot as plt

# 程序中链接了多个OpenMP运行时库副本，此处作为临时解决方案
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# noinspection PyPep8Naming,PyShadowingNames
def evaluate_accuracy_gpu(net, data_iter, device=None):
    # 使用GPU计算模型在数据集上的精度
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        # 如果未指定device，函数会自动选择模型参数所在的设备
        if not device:
            # iter将生成器转换为一个迭代器
            # next返回模型的第一个参数张量
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量，两个变量
    metric = d2l.Accumulator(2)
    # 关闭梯度计算
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的
                # 将列表中的每个张量都移动到指定设备
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            # 计算准确性并累加到累加器中
            # y.numel()返回张量y中包含的元素总数
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# noinspection PyPep8Naming,PyShadowingNames
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    # 用GPU训练模型
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            # Xavier均匀初始化
            nn.init.xavier_uniform_(m.weight)
    # 将初始化函数应用到模型的所有层上
    net.apply(init_weights)
    # 确认训练设备
    print('training on', device)
    net.to(device)
    # 随机梯度下降优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # 交叉熵损失函数
    loss = nn.CrossEntropyLoss()
    # 可视化
    animator = d2l.Animator(xlabel='epoch',
                            xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    # d2l.Timer用于记录训练时间
    timer, num_batches = d2l.Timer(), len(train_iter)
    train_l, train_acc, test_acc, metric = 0.0, 0.0, 0.0, 0.0
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()  # 训练模式
        # enumerate将一个可迭代对象转换为一个迭代器
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()  # 梯度清零
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()  # 反向传播
            optimizer.step()  # 参数更新
            with torch.no_grad():
                # 总损失，准确率，样本数量
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            # 可视化更新
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1 / num_batches, (None, None,  test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    # 每秒处理的样本数
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
# 查看各个层
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

lr = 0.9
num_epochs = 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

plt.show()
