from torch_geometric.nn import GAE, VGAE, GCNConv
from utils import *
from torch.nn.functional import normalize
from abl_wo_disen import Encoder
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import argparse
import sys
import time
from matplotlib import patches

def get_args():
    parser = argparse.ArgumentParser(description='CFLP')
    parser.add_argument('--dataset', type=str, default='Synthetic dataset')
    parser.add_argument('--num_c', type=int, default=500, help='node numbers for every community')
    parser.add_argument('--q', type=float, default=0.000001, help='probability of linking between communities')

    parser.add_argument('--p1', type=float, default=0.01, help='probability of linking within 1st communities')
    parser.add_argument('--p2', type=float, default=0.02, help='probability of linking within 2nd communities')
    parser.add_argument('--p3', type=float, default=0.05, help='probability of linking within 3rd communities')
    parser.add_argument('--p4', type=float, default=0.04, help='probability of linking within 4th communities')
    parser.add_argument('--p5', type=float, default=0.08, help='probability of linking within 4th communities')

    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--name', type=str, default='debug', help='name for this run for logging')

    parser.add_argument('--model', type=str, default='VGAE')
    parser.add_argument('--k', type=int, default=5, help='channels')
    parser.add_argument('--x_dim', type=int, default=8, help='dimension of each channels')
    parser.add_argument('--routit', type=int, default=3, help='iteration of disentangle')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--gpu', type=int, default=-1)

    parser.add_argument('--val_frac', type=float, default=0.05, help='fraction of edges for validation set')
    parser.add_argument('--test_frac', type=float, default=0.1, help='fraction of edges for testing set')
    args = parser.parse_args()
    args.argv = sys.argv

    args.device = torch.device('cuda:0' if args.gpu >= -1 else 'cpu')

    return args

#compute person
def comput_cor(args):
    _, _, _, data = dataloader(args)
    model = eval(args.model)(encoder=Encoder(data.x.size(1), args)).to(args.device)
    model.load_state_dict(torch.load('syn_raw.pth'))
    z=model.encode(data.x, args)
    # z=normalize(z)
    z=z.detach().cpu().numpy()
    np.savetxt("embedding.csv", z, delimiter=',')
    csv_data = pd.read_csv("./embedding.csv")
    cor=csv_data.corr(method='pearson')
    return abs(cor)

def cor_fig(cor, args):
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['font.family'] = 'Times New Roman'

    label=np.arange(0, args.k*args.x_dim+1, args.x_dim).tolist()
    ax = sns.heatmap(cor, vmin=0.0, vmax=1.0, cmap="YlGnBu")
    ax.set_xticks(np.arange(0, args.k*args.x_dim+1, args.x_dim), label)
    ax.set_yticks(np.arange(0, args.k*args.x_dim+1, args.x_dim), label)

    rect1 = patches.Rectangle(xy=(0, 0), width=args.x_dim, height=args.x_dim, linewidth=1.8, linestyle='--',
                              edgecolor='r', facecolor='none')
    ax.add_patch(rect1)
    rect2 = patches.Rectangle(xy=(args.x_dim, args.x_dim), width=args.x_dim, height=args.x_dim, linewidth=1.8,
                              linestyle='--', edgecolor='r', facecolor='none')
    ax.add_patch(rect2)
    rect3 = patches.Rectangle(xy=(2 * args.x_dim, 2 * args.x_dim), width=args.x_dim, height=args.x_dim, linewidth=1.8,
                              linestyle='--', edgecolor='r', facecolor='none')
    ax.add_patch(rect3)
    rect4 = patches.Rectangle(xy=(3 * args.x_dim, 3 * args.x_dim), width=args.x_dim, height=args.x_dim, linewidth=1.8,
                              linestyle='--', edgecolor='r', facecolor='none')
    ax.add_patch(rect4)
    rect5 = patches.Rectangle(xy=(4 * args.x_dim, 4 * args.x_dim), width=args.x_dim, height=args.x_dim, linewidth=1.8,
                              linestyle='--', edgecolor='r', facecolor='none')
    ax.add_patch(rect5)

    plt.savefig('raw_feat.pdf', bbox_inches='tight', pad_inches=0.1, dpi=1500)
    plt.savefig('raw_feat.png', bbox_inches='tight', pad_inches=0.1, dpi=1500)

def main(args):
    # seed = 717
    # set_rng_seed(seed)
    log_name = f'{args.log_dir}/{args.name}_{args.dataset}_{args.model}_{time.strftime("%Y-%m-%d,%H:%M")}'
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(log_name)
    logger.info(f'args: {args}')
    # logger.info(f'seed:{seed}')
    cor = comput_cor(args)
    cor_fig(cor, args)

if __name__ == "__main__":
    args = get_args()
    main(args)





