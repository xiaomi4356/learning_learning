import random
import sys
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import torch
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model import Disen_Linear
from torch_geometric.nn import GAE, VGAE, GCNConv

parser = argparse.ArgumentParser(description='DGAE')
parser.add_argument('--k', type=int, default=5, help='channels')
parser.add_argument('--x_dim', type=int, default=64, help='dimension of each channels')
parser.add_argument('--routit', type=int, default=3, help='iteration of disentangle')
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--model', type=str, default='VGAE')

args = parser.parse_args()
args.argv = sys.argv
args.device = torch.device('cuda:1')

################################################################################
transform = T.Compose([T.NormalizeFeatures(), T.ToDevice(args.device)])
dataset_cora = Planetoid(root='../dataset', name='Cora', transform=transform)
data = dataset_cora[0]
model = eval(args.model)(encoder=Disen_Linear(data.x.size(1), args)).to(args.device)
model.load_state_dict(torch.load('Cora_disen.pth'))
model.eval()
with torch.no_grad():
    z = model.encode(data.x, data.edge_index, args)
    z_tsne = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())
    z_min, z_max = z_tsne.min(0), z_tsne.max(0)
    z_norm = (z_tsne - z_min) / (z_max - z_min)
##############################################################################################
plt.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 8})
plt.figure(figsize=(5,5))
# fig.tight_layout()  # 调整整体空白

plt.scatter(z_norm[:, 0], z_norm[:, 1], s=2, c=data.y.detach().cpu().numpy(), cmap="Set2")
plt.xticks([])
plt.yticks([])
plt.axis('off')

plt.savefig("disen_tsne.pdf", bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.savefig("disen_tsne.png", bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.show()
