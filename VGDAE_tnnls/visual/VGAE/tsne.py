import random
import sys
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import torch
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from main_VGAE import VariationalGCNEncoder
from torch_geometric.nn import GAE, VGAE, GCNConv

parser = argparse.ArgumentParser(description='DGAE')
parser.add_argument('--model', type=str, default='VGAE')
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--out_dim', type=int, default=16)
args = parser.parse_args()
args.argv = sys.argv
args.device = torch.device('cuda:1')

################################################################################
transform = T.Compose([T.NormalizeFeatures(), T.ToDevice(args.device)])
dataset_cora = Planetoid(root='../dataset', name='Cora', transform=transform)
data = dataset_cora[0]
model = eval(args.model)(VariationalGCNEncoder(data.x.size(1), args.out_dim)).to(args.device)
model.load_state_dict(torch.load('Cora_vgae.pth'))
z = model.encode(data.x, data.edge_index)
model.eval()
with torch.no_grad():
    z = model.encode(data.x, data.edge_index)
    z_tsne= TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())
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

plt.savefig("vgae_tsne.pdf", bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.savefig("vgae_tsne.png", bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.show()
