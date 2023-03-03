import torch
from torch import nn

batch_size = 2
hidden_size = 16
input_size = 7
num_layers = 3
seq_len = 1

model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

input_tensor = torch.rand(seq_len, batch_size, input_size)

out, (h, c) = model(input_tensor)
out, (h, c) = model(input_tensor, (None, None))
print(out.shape)
print(h.shape)
print(c.shape)
