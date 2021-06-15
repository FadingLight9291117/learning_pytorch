import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 建立数据集

# 随机数据集
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

# 用 Varibale 来修饰这些数据 tensor
x, y = Variable(x), Variable(y)

plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()


# 建立神经网络

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()

        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)

print(net)  # net 的结构


# train
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # 随机梯度下降， lr学习率
loss_func = torch.nn.MSELoss()  # 均方差损失函数

plt.ion()
plt.show()

prediction = None

for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)  # 计算两者误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播计算参数更新值
    optimizer.step()        # 将更新的值施加到 net 的 parameter 上

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data,
                 fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

print(prediction.data.numpy())
