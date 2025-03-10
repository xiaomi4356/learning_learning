import torch.nn as nn
import torch.nn.functional as F
import torch

# Feature disentangle layer
class RoutingLayer(nn.Module):
    def __init__(self, k, routit):

        super(RoutingLayer, self).__init__()
        self.k = k
        self.routit = routit

    def forward(self, x, src_trg):
        m, src, trg = src_trg.shape[1], src_trg[0], src_trg[1]
        n, d = x.shape
        k, delta_d = self.k, d // self.k

        x = F.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = x[src].view(m, k, delta_d)  # neighbors' feature
        c = x  # node-neighbor attention aspect factor
        for t in range(self.routit):
            p = (z * c[trg].view(m, k, delta_d)).sum(dim=2)  # update node-neighbor attention aspect factor
            p = F.softmax(p, dim=1)
            p = p.view(-1, 1).repeat(1, delta_d).view(m, k, delta_d)

            weight_sum = (p * z).view(m, d)  # weight sum (node attention * neighbors feature)
            c = c.index_add_(0, trg, weight_sum)  # update output embedding
            c = F.normalize(c.view(n, k, delta_d), dim=2).view(n, d) # embedding normalize aspect factor
        return c

class Disen_Linear(nn.Module):
    def __init__(self, in_dim, k, x_dim, model, routit):
        super(Disen_Linear, self).__init__()
        self.linear = nn.Linear(in_dim, k*x_dim)
        if model == 'VGAE':
            self.linear_= nn.Linear(in_dim, k*x_dim)
        self.routlay = RoutingLayer(k, routit)


    def forward(self, x, edge_index, model):
        if model=='GAE':
            x = self.linear(x)
            x = self.routlay(x, edge_index)
            return x

        if model=='VGAE':
            mu = self.linear(x)
            mu = self.routlay(mu, edge_index)
            log = self.linear_(x)
            return mu, log



class max_deco(nn.Module):
    def   __init__(self, hyperpm):
        super(max_deco, self).__init__()
        self.k = hyperpm['k']
        self.d = hyperpm['x_dim']

    def forward(self, z, edge_index, sigmoid=True):
        m= edge_index.size(1)
        src = z[edge_index[0]].view(m, self.k, self.d)
        trg = z[edge_index[1]].view(m, self.k, self.d)
        p = (src*trg).sum(dim=2)
        value = torch.max(p, dim=1, keepdim=False).values

        return torch.sigmoid(value) if sigmoid else value


