import sys
from model import DisenEncoder
from utils import *
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

seed=random.randint(1,100000)
print(seed)
set_rng_seed(seed)
parser = argparse.ArgumentParser(description='CDG')
parser.add_argument('--k', type=int, default=5, help='channels')
parser.add_argument('--x_dim', type=int, default=32, help='dimension of each channels')
parser.add_argument('--routit', type=int, default=4, help='iteration of disentangle')

args = parser.parse_args()
args.argv = sys.argv

transform = T.Compose([T.NormalizeFeatures()])
dataset = Planetoid(root='../dataset', name='Pubmed', transform=transform)
data= dataset[0]

model = DisenEncoder(data.x.size(1), args)
model.load_state_dict(torch.load('best_cdg_Pubmed_tsne.pth'))
model.eval()
with torch.no_grad():
    z = model(data.x, data.edge_index)
    z_d = TSNE(n_components=2, perplexity=300).fit_transform(z.detach().cpu().numpy())


plt.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 8})

plt.figure(figsize=(4,4))
plt.xticks([])
plt.yticks([])

plt.scatter(z_d[:, 0], z_d[:, 1], s=2, c=data.y, cmap="Set2")
plt.title("Embeddings by CLDGE", y=-0.1)
plt.axis('off')
plt.savefig("demo.png", bbox_inches='tight', pad_inches=0.1,  dpi=600)
plt.show()