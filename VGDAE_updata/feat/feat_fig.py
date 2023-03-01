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
import numpy as np
import seaborn as sns
from matplotlib import patches

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
hyperpm=dict()
hyperpm['dataset']='Cora'
hyperpm['alpha']=0.01
hyperpm['k']=5
hyperpm['x_dim']=5
hyperpm['routit']=3
hyperpm['dropout']=0
hyperpm['model']='GAE'
hyperpm['mi_estimators']='CLUBSample'
hyperpm['seed'] = 358
set_rng_seed(hyperpm['seed'])

print(f'================================================================')
# log_param(hyperpm)
train_data, val_data, test_data = dataloader(hyperpm, device)
print(train_data)
model = eval(hyperpm['model'])(encoder=Disen_Linear(train_data.x.size(1), hyperpm)).to(device)
print(model)
num_dis = int(hyperpm['k'] * (hyperpm['k'] - 1) / 2)
mi_estimators = ModuleList([eval(hyperpm['mi_estimators'])(hyperpm['x_dim'], hyperpm['x_dim'], hyperpm['x_dim']) for i in range(num_dis)]).to(device)

model.load_state_dict(torch.load('cora_feat.pth'))
z=model.encode(train_data.x, train_data.edge_index, hyperpm)
z=z.detach().numpy()
# print(z)
#compute person
cor = np.zeros((z.shape[1], z.shape[1]))
for i in range(z.shape[1]):
    for j in range(z.shape[1]):
        cof = scipy.stats.pearsonr(z[:, i], z[:, j])[0]
        # cof = scipy.stats.kendalltau(z[:, i], z[:, j])[0]
        cor[i][j] = cof

print(cor)
def plot_corr(data):

    config = {
        "font.family": 'serif',  # sans-serif/serif/cursive/fantasy/monospace
        "font.size": 12,  # medium/large/small
        'font.style': 'normal',  # normal/italic/oblique
        'font.weight': 'normal',  # bold
        "mathtext.fontset": 'cm',  # 'cm' (Computer Modern)
        "font.serif": ['Times New Roman'],  # 'Simsun'宋体
        "axes.unicode_minus": False,  # 用来正常显示负号
    }
    plt.rcParams.update(config)

    ax = sns.heatmap(data, vmin=0.0, vmax=1.0, cmap="YlGnBu")

    rect1 = patches.Rectangle(xy=(0,0), width=5, height=5, linewidth=1.7, linestyle='--', edgecolor='r', facecolor='none')
    ax.add_patch(rect1)
    rect2 = patches.Rectangle(xy=(5,5), width=5, height=5, linewidth=1.7, linestyle='--', edgecolor='r', facecolor='none')
    ax.add_patch(rect2)
    rect3 = patches.Rectangle(xy=(10, 10), width=5, height=5, linewidth=1.7, linestyle='--', edgecolor='r', facecolor='none')
    ax.add_patch(rect3)
    rect4 = patches.Rectangle(xy=(15, 15), width=5, height=5, linewidth=1.7, linestyle='--', edgecolor='r', facecolor='none')
    ax.add_patch(rect4)
    rect5 = patches.Rectangle(xy=(20, 20), width=5, height=5, linewidth=1.7, linestyle='--', edgecolor='r', facecolor='none')
    ax.add_patch(rect5)

    plt.savefig('club_feat_fig.eps', bbox_inches='tight', pad_inches=0.1, dpi=800)
    plt.savefig('club_feat_fig.png', bbox_inches='tight', pad_inches=0.1, dpi=800)
    plt.close()
plot_corr(np.abs(cor))

