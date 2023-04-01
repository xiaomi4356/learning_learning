from torch_geometric.nn import GAE, VGAE, GCNConv
from model import Disen_Linear
from utils import *
from correlation import DistanceCorrelation
from torch import round
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hyperpm=dict()
hyperpm['dataset']='Cora'
hyperpm['k']=6
hyperpm['hidden_dim']=64
hyperpm['x_dim']=64
hyperpm['mi_iter']=5
hyperpm['routit']=3
hyperpm['dropout']=0
hyperpm['model']='VGAE'

train_data, val_data, test_data = dataloader(hyperpm, device)
# #VGDAE
model = eval(hyperpm['model'])(encoder=Disen_Linear(train_data.x.size(1), hyperpm)).to(device)
model.load_state_dict(torch.load('vgdae_cora.pth'))
z = model.encode(train_data.x, train_data.edge_index, hyperpm)
print(z.shape)
cor_list=[]
for i in range(hyperpm['k']):
    for j in range(hyperpm['k']):
        cor = DistanceCorrelation(z[:, i * hyperpm['x_dim']: (i + 1) * hyperpm['x_dim']],
                                      z[:, j * hyperpm['x_dim']: (j + 1) * hyperpm['x_dim']])
        cor = round(cor, decimals=3)
        cor_list.append(cor)
cor_arr = np.array(cor_list).reshape(6,6)
print(cor_arr)


