import random
import sys
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import torch
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser(description='DGAE')
parser.add_argument('--k', type=int, default=5, help='channels')
parser.add_argument('--x_dim', type=int, default=32, help='dimension of each channels')
parser.add_argument('--routit', type=int, default=4, help='iteration of disentangle')

args = parser.parse_args()
args.argv = sys.argv

args.device = torch.device('cuda:1')
################################################################################
transform = T.Compose([T.NormalizeFeatures()])
dataset_cora = Planetoid(root='../dataset', name='Cora', transform=transform)
data_cora = dataset_cora[0]
x_cora = TSNE(n_components=2, perplexity=20).fit_transform(data_cora.x.detach().cpu().numpy())
z_min, z_max = x_cora .min(0), x_cora .max(0)
z_norm = (x_cora - z_min) / (z_max - z_min)
##############################################################################################
plt.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 8})
plt.figure(figsize=(5,5))
# fig.tight_layout()  # 调整整体空白


plt.scatter(z_norm[:, 0], z_norm[:, 1], s=2, c=data_cora.y, cmap="Set2")
plt.xticks([])
plt.yticks([])
plt.axis('off')

plt.savefig("raw_tsne.pdf", bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.savefig("raw_tsne.png", bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.show()
