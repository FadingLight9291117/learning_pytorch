import torch
import torch.nn.functional as F
from torch.autograd import Variable

# 一些数据
x = torch.linspace(-5, 5, 200)
x = Variable(x)

x_np = x.data.numpy()

# 几种激活函数
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_relu = F.relu(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
# y_softmax = F.softmax(x)


