import torch
from torch import nn
class max_deco(nn.Module):
    def   __init__(self, hyperpm):
        super(max_deco, self).__init__()
        self.k = hyperpm['k']
        self.d = hyperpm['out_dim']//hyperpm['k']

    def forward(self, z, edge_index, sigmoid=True):
        m= edge_index.size(1)
        src = z[edge_index[0]].view(m, self.k, self.d)
        trg = z[edge_index[1]].view(m, self.k, self.d)
        p = (src*trg).sum(dim=2)
        value = torch.max(p, dim=1, keepdim=False).values

        return torch.sigmoid(value) if sigmoid else value


