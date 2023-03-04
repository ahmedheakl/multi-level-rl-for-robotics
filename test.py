import torch

output = []
for _ in range(3):
    output.append(torch.rand(1, 3))


x = torch.concat(output, dim=1)
print(x.shape)
