from torch_geometric.nn import GAE, VGAE, GCNConv
from model import Disen_Linear
from utils import *
import numpy as np
import torch
from torch.nn.functional import cosine_similarity as cos
import torch.nn.functional as F
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
hyperpm=dict()
hyperpm['dataset']='Cora'
hyperpm['k']=4
hyperpm['hidden_dim']=64
hyperpm['x_dim']=64
hyperpm['mi_iter']=5
hyperpm['routit']=3
hyperpm['dropout']=0
hyperpm['model']='VGAE'
set_rng_seed(1376585645)
train_data, val_data, test_data = dataloader(hyperpm, device)
# #VGDAE
model = eval(hyperpm['model'])(encoder=Disen_Linear(train_data.x.size(1), hyperpm)).to(device)
# model.load_state_dict(torch.load('vgdae_cora.pth'))
model.load_state_dict(torch.load('vgdae_cora.pth'))

z = model.encode(train_data.x, train_data.edge_index, hyperpm)
N=z.size(0)
print(N)
print(z.shape)
cor_l=torch.zeros(size=(1,N))
cor_c=torch.zeros(size=(hyperpm['k'], hyperpm['k']))
for i in range(hyperpm['k']):
    for j in range(hyperpm['k']):
        x1 = z[:, i * hyperpm['x_dim']: (i + 1) * hyperpm['x_dim']]
        x2 = z[:, j * hyperpm['x_dim']: (j + 1) * hyperpm['x_dim']]
        for m in range(N):
            cor = cos(x1[m, :], x2).norm(p=1)
            cor_l[0, m] = cor.item()
        # print(cor_l.size())
        cor_cos =1/(N*N)*(cor_l.norm(p=2))
        cor_c[i,j]=cor_cos
cor_c=F.normalize(cor_c, dim=1)
print(cor_c)


