from torch_geometric.nn import GAE, VGAE, GCNConv, global_mean_pool
from model import Disen_Linear
from utils import *
import numpy as np
import torch
from torch.nn.functional import cosine_similarity as cos
import torch.nn.functional as F
from torch.nn import Module, Linear
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hyperpm=dict()
hyperpm['dataset']='Cora'
hyperpm['k']=4
hyperpm['hidden_dim']=64
hyperpm['x_dim']=64
hyperpm['mi_iter']=5
hyperpm['routit']=3
hyperpm['dropout']=0
hyperpm['model']='VGAE'
set_rng_seed(78)
train_data, val_data, test_data = dataloader(hyperpm, device)
# #VGDAE
class Encoder(Module):
    def __init__(self, input, hyperpm):
        super(Encoder, self).__init__()
        self.linear = Linear(in_features=input, out_features=hyperpm['k']*hyperpm['x_dim'])
        if hyperpm['model'] == 'VGAE':
            self.linear_ = Linear(in_features=input, out_features=hyperpm['k']*hyperpm['x_dim'])
    def forward(self, x):
        if hyperpm['model']=='GAE':
            x = self.linear(x)
            return x

        if hyperpm['model']=='VGAE':
            mu = self.linear(x)
            log = self.linear_(x)
            return mu, log


model = eval(hyperpm['model'])(encoder=Encoder(train_data.x.size(1), hyperpm)).to(device)
model.load_state_dict(torch.load('wo_disen_cora.pth'))
z = model.encode(train_data.x)
# model = eval(hyperpm['model'])(encoder=Disen_Linear(train_data.x.size(1), hyperpm)).to(device)
# model.load_state_dict(torch.load('wo_club_cora.pth'))
# z = model.encode(train_data.x, train_data.edge_index, hyperpm)
N=z.size(0)
print(N)
print(z.shape)
cor_l=torch.zeros(hyperpm['k'], hyperpm['k'])

for i in range(hyperpm['k']):
    for j in range(hyperpm['k']):
        x1 = z[:, i * hyperpm['x_dim']: (i + 1) * hyperpm['x_dim']]
        x2 = z[:, j * hyperpm['x_dim']: (j + 1) * hyperpm['x_dim']]
        print(x1.shape)
        x1, x2 = x1.sum(dim=0).view(1, 64), x2.sum(dim=0).view(1, 64)
        print(x1.shape)
        cor = cos(x1, x2)
        cor_l[i,j] = cor.item()
cor_l=torch.triu(cor_l,diagonal=1)
mask=cor_l!=0
cor_l=F.normalize(cor_l[mask], dim=1)
cor_l=F.normalize(cor_l, dim=0)
print(f'wo_club_cora.pth={cor_l}')

