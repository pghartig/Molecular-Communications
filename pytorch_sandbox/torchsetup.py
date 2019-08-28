from __future__ import print_function
import torch

x = torch.empty(2,2, requires_grad=True)
print(x)

y = x+2
print(y)








