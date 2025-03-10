import torch.nn as nn
import torch.nn.functional as F

# Feature disentangle layer
class RoutingLayer(nn.Module):
    def __init__(self, hyperpm):
        super(RoutingLayer, self).__init__()
        self.k = hyperpm['k']
        self.routit = hyperpm['routit']

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
    def __init__(self, in_dim, hyperpm):
        super(Disen_Linear, self).__init__()
        self.linear = nn.Linear(in_dim, hyperpm['out_dim'])
        self.routlay = RoutingLayer(hyperpm)
        self.dropout = hyperpm['dropout']


    def forward(self, x, edge_index):
        x = self.linear(x)
        x = self.routlay(x, edge_index)



        return x
