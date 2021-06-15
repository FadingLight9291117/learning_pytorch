import numpy as np
import torch

data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)

# 矩阵点乘
print(
        "\nmatrics multipication",
        "\nnumpy: ", np.matmul(data, data), 
        "\ntorch: ", tensor.mm(tensor)
        )


