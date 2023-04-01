import sys

import numpy as np
import torch
import torch.nn as nn
from model import DisenEncoder, pretext_loss, LogReg, acc
from utils import *
from torch_geometric.utils import dropout_adj
import time
import argparse
from eval import label_classification
def get_args():
    parser = argparse.ArgumentParser(description='CDG')
    parser.add_argument('--dataset', type=str, default='reddit')
    parser.add_argument('--datapath', type=str, default='../dataset')
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--name', type=str, default='debug', help='name for this run for logging')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--enc_lr', type=float, default=0.01, help='the learning rate of total model')
    parser.add_argument('--enc_weight_decay', type=float, default=0.0001, help='weight_decay of total model')
    parser.add_argument('--k', type=int, default=4, help='channels')
    parser.add_argument('--x_dim', type=int, default=64, help='dimension of each channels')
    parser.add_argument('--routit', type=int, default=4, help='iteration of disentangle')
    parser.add_argument('--gpu', type=int, default=-1)

    parser.add_argument('--de_rate1', type=float, default=0.3, help='dropout edges rate for view1')
    parser.add_argument('--de_rate2', type=float, default=0.2, help='dropout edges rate for view2')
    parser.add_argument('--df_rate1', type=float, default=0.3, help='dropout features rate for view1')
    parser.add_argument('--df_rate2', type=float, default=0.5, help='dropout features rate for view2')
    parser.add_argument('--m', type=int, default=20, help='inter negative samples')
    parser.add_argument('--n', type=int, default=4, help='intra negative samples')
    parser.add_argument('--log_epoch', type=int, default=200, help='epoch for logreg')
    parser.add_argument('--log_lr', type=float, default=0.01, help='the learning rate of LogReg for classification')
    parser.add_argument('--log_weight_decay', type=float, default=0.0001, help='weight_decay of LogReg classification')
    parser.add_argument('--log_trails', type=int, default=10)
    parser.add_argument('--seed', type=int, default=-1, help='fix random seed if needed')
    parser.add_argument('--ratio', type=float, default=0.8, help='ratio for train set')

    args = parser.parse_args()
    args.argv = sys.argv

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    args.device = torch.device('cuda' if args.gpu >= -1 else 'cpu')

    return args


def Encoder(args, logger):

    # load data
    data, nclass = dataloader(args)
    #generate two views
    edge_index_1 = dropout_adj(data.edge_index, p=args.de_rate1)[0]
    edge_index_2 = dropout_adj(data.edge_index, p=args.de_rate2)[0]
    x_1 = drop_feature(data.x, args.df_rate1)
    x_2 = drop_feature(data.x, args.df_rate2)
    #model and optim
    model = DisenEncoder(data.x.size(1), args).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.enc_lr, weight_decay=args.enc_weight_decay)

    #train
    best_loss, step = 1e6, 0
    for epoch in range(args.epoch):
        model.train()
        optim.zero_grad()
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        loss = pretext_loss(z1, z2, args.k, args.n, args.m)

        if loss < best_loss:
            best_loss = loss
            step = epoch
            torch.save(model.state_dict(), 'best_cdg.pth')

        loss.backward()
        optim.step()
        logger.info(f'Epoch:{epoch:03d}, loss:{loss:.4f}')
    logger.info(f'step:{step:03d}, best_loss:{best_loss:.4f}')

    # obtain embeddings
    model.load_state_dict(torch.load('best_cdg.pth'))
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

    return z, data.y

def main(args):
    log_name = f'{args.log_dir}/{args.name}_{args.dataset}_{time.strftime("%Y-%m-%d,%H:%M")}'
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(log_name)
    logger.info(f'args: {args}')
    if args.seed > 0:
        set_rng_seed(args.seed)
    logger.info(f'================================Encoder run=================================')
    z, y = Encoder(args, logger)
    logger.info(f'================================Encoder run ends=================================')
    test_acc_list = []
    for i in range(args.log_trails):
        logger.info(f'================================Log runs {i+1}=================================')
        test_acc = label_classification(z, y, args.ratio, logger)
        test_acc_list.append(test_acc)
        logger.info(f'model runs {i + 1} times, test_acc:{test_acc:.4f}')
    logger.info(f'model runs times, mean_acc={np.mean(test_acc_list)}+-{np.std(test_acc_list)}')
    logger.info(f'================================Log runs ends=================================')

if __name__ == "__main__":
    args = get_args()
    main(args)
