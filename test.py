import torch
import numpy as np

a = torch.tensor([[1], [2], [3]])
b = torch.tensor([[5], [6], [7]])
c = torch.tensor([[9], [4], [0]])

f = a.mul_(b)




d = torch.stack([a, b, c])
e = torch.cat([a, b, c])

l = [a.numpy(), b.numpy()]

l.append(c.numpy())

l = np.array(l).flatten()

x = torch.cat(l)
y = torch.stack(l)

print(l)

print(d.size())
print(e.size())