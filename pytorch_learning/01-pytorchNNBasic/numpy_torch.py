import numpy as np
import torch

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

print(
        '\nnumpy: \n', np_data,
        '\ntorch: \n', torch_data,
        '\ntensor to array: \n', tensor2array
        )
