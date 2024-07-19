import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt

d2l.use_svg_display()


# 文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 绘制图像
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            img = img.squeeze().numpy()  # 将图像数据从 (1, 28, 28) 变为 (28, 28)
        ax.imshow(img, cmap='gray')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()


def get_dataloader_workers():
    return 4


trans = transforms.ToTensor()  # 将图像数据从PIL类型变换成32位浮点数格式
# 下载数据
mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)

print(len(mnist_train), len(mnist_test))
print(mnist_train[0][0].shape)

x, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(x.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())

# 取训练数据所需的时间
timer = d2l.Timer()
for x, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')
