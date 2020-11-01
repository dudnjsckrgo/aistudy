import torch
t = torch.rand(4, 4)
b = t.view(-1)
print(t)
print(b)