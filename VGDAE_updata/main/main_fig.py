import random
import torch
from torch_geometric.nn import GAE, VGAE, GCNConv
from model import Disen_Linear
from model import max_deco
from utils import *
from correlation import *
from torch.nn import ModuleList
import numpy as np
import scipy

import matplotlib.pyplot as plt
import matplotlib.axes as ax
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors


device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
hyperpm=dict()
hyperpm['dataset']='Citeseer'
hyperpm['alpha']=0.01
hyperpm['k']=8
hyperpm['x_dim']=8
hyperpm['routit']=3
hyperpm['dropout']=0
hyperpm['model']='VGAE'
hyperpm['mi_estimators']='CLUBSample'
hyperpm['seed'] = 8093
set_rng_seed(hyperpm['seed'])

print(f'================================================================')
# log_param(hyperpm)
train_data, val_data, test_data = dataloader(hyperpm, device)
print(train_data)
model = eval(hyperpm['model'])(encoder=Disen_Linear(train_data.x.size(1), hyperpm)).to(device)
print(model)
num_dis = int(hyperpm['k'] * (hyperpm['k'] - 1) / 2)
mi_estimators = ModuleList([eval(hyperpm['mi_estimators'])(hyperpm['x_dim'], hyperpm['x_dim'], hyperpm['x_dim']) for i in range(num_dis)]).to(device)

model.load_state_dict(torch.load('club_gae_citeseer.pth'))
z=model.encode(train_data.x, train_data.edge_index, hyperpm)
z=z.detach().numpy()
# print(z)
#compute person
cor = np.zeros((z.shape[1], z.shape[1]))
for i in range(z.shape[1]):
    for j in range(z.shape[1]):
        # cof = scipy.stats.pearsonr(z[:, i], z[:, j])[0]
        cof = scipy.stats.kendalltau(z[:, i], z[:, j])[0]
        cor[i][j] = cof

print(cor)
def plot_corr(data):
    ax = sns.heatmap(data, vmin=0.0, vmax=1.0, cmap="YlGn")
    plt.savefig('feat_fig.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
    plt.savefig('feat_fig.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
    plt.close()
plot_corr(np.abs(cor))

