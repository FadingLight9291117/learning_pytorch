"""
优化器 optimizer
1. SGD
2. Momentum
3. RMSprop
4. Adam
"""
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)

LR = 0.1
BATCH_SIZE = 32
EPOCH = 12

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.05 * torch.normal(torch.zeros(*x.size()))

# plot data
plt.scatter(x.numpy(), y.numpy())
plt.show()

torch_data = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.output = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x


net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.09))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()
losses = [[], [], [], []]

for epoch in range(EPOCH):
    print(f'Epoch: {epoch}')
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        for net, opt, lo in zip(nets, optimizers, losses):
            output = net(b_x)
            loss = loss_func(output, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lo.append(loss.data.numpy())

# 画图
labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, lo in enumerate(losses):
    plt.plot(lo, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()
