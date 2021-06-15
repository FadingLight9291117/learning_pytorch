import torch
from torch.autograd import Variable

data = [[1, 2], [3, 4]]

tensor = torch.FloatTensor(data)

variable = Variable(tensor, requires_grad=True)

print(tensor)
print(variable)

v_out = torch.mean(variable * variable)

print(v_out)

v_out.backward() # 误差反向传播

print(v_out.grad) # 输出Varibale的梯度


# 获取Variable里面的数据
print(variable.data)
print(variable.data.numpy())
