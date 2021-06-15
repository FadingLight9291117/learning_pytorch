"""
数据读取 DataLoader的使用
"""

import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt

torch.manual_seed(1)

BATCH_SIZE = 4

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)
for epoch in range(3):  # 训练所有!整套!数据 3 次
    for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
        # 假设这里就是你训练的地方...

        # 打出来一些数据
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())

        plt.scatter(batch_x.numpy(), batch_y.numpy())
        plt.show()