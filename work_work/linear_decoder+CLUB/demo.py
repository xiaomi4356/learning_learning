import torch
m=torch.randint(0,20,(4,5))
n=m[:, :3*2]
print(m)
print(n)