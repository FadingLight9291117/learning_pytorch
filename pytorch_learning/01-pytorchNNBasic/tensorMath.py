import tensor
import numpy as np

data = [-1, 2, -2, 5]
tensor = torch.FloatTensor(data)

# abs绝对值
print(
        "\nabs",
        "\nnumpy: ", np.abs(data),
        "\ntorch: ", torch.abs(data),
        )

# sin 三角函数
print(
        "\nsin",
        "\nnumpy: ", np.sin(data),
        "\ntorch: ", torch.sin(data)
        )

# mean 均值
print(
        "\nmean",
        "\nnumpy: ", np.mean(data),
        "\ntorch: ", torch.mean(data)
        )

