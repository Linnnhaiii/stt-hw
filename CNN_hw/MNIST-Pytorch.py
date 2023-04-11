import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

train_set = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_set = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
# print(train_set.data.size())

# 卷积神经网络的基本模块，卷积模块，包含卷积层和激活函数
# attention: 如果换成其他的卷积核尺寸，则需要重新计算padding，使得模型的输出（后续全连接层的输入）维度匹配
# 也可以考虑把padding和stride都放到init函数的参数里
class Conv_block(nn.Module):
    def __init__(self, ks, ch_in, ch_out):
        super(Conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=ks, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class CNN(nn.Module):
    def __init__(self, kernel_size, in_ch):
        super(CNN, self).__init__()
        outch_list = [16, 32, 64, 128, 256] # 每一层卷积层的输出特征图的维度数，也可以直接写，不用这样
        self.conv1 = Conv_block(kernel_size, in_ch, outch_list[0])
        self.conv2 = Conv_block(kernel_size, outch_list[0], outch_list[1])
        self.conv3 = Conv_block(kernel_size, outch_list[1], outch_list[2])
        self.conv4 = Conv_block(kernel_size, outch_list[2], outch_list[3])
        self.conv5 = Conv_block(kernel_size, outch_list[3], outch_list[4])
        # 全连接层，用来分类
        self.fc = nn.Sequential(
            nn.Linear(outch_list[4] * 28 * 28, 1024), # 将特征图展平
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        conv_output = self.conv1(x)
        conv_output = self.conv2(conv_output)
        conv_output = self.conv3(conv_output)
        conv_output = self.conv4(conv_output)
        conv_output = self.conv5(conv_output)
        
        fc_in = conv_output.view(conv_output.size()[0], -1) # n, c, h, w ===> n, c*h*w
        
        output = self.fc(fc_in)

        return output

# 定义超参数
learning_rate = 1e-3
batch_size = 64
epochs = 20

# 定义dataloader，每一次迭代会返回一个batch大小的样本以及对应的标签
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

model = CNN(3, 1).to(device)

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(dataloader, model, loss_fn, optimizer):
    for X, y in dataloader:
        # forward
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))
        # back prop.
        optimizer.zero_grad() # 重置模型参数的梯度，默认情况下梯度会迭代相加
        loss.backward() # 将loss反向传播
        optimizer.step() # 梯度下降，W = W - lr * 梯度

    print(f"Training loss is {loss.item()}")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset) # 对应的数据集里一共多少个样本
    num_batches = len(dataloader) # batch_size
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y.to(device)).item()
            # 算有多少个预测对了
            correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()
        
    test_loss /= num_batches
    correct /= size

    print(f"Testing loss: {test_loss}, Accuracy: {100 * correct}% \n")

if __name__ == "__main__":
    for e in range(epochs):
        print(f"Epoch {e+1} \n --------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    torch.save(model, 'cnn_model.pth')