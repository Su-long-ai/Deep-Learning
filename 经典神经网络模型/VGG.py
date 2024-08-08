import torch
from torch import nn
from d2l import torch as d2l
import os
import matplotlib.pyplot as plt

# 我的程序中链接了太多个OpenMP运行时库副本，此处作为临时解决方案，不推荐
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def vgg_block(num_convs, in_channels, out_channels):
    # 初始化一个空列表
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# noinspection PyShadowingNames
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    out_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        # 将conv_blks列表解包成单独的参数传递给nn.Sequential
        *conv_blks,
        nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

net = vgg(conv_arch)

X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)

# 缩放比例，用于减小网络的规模
ratio = 4
# 根据缩放比例调整每个卷积块的输出通道数
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

plt.show()
