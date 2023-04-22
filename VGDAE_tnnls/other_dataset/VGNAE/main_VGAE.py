import random
import torch
import torch.nn as nn
import sys
from torch_geometric.nn import GAE, VGAE, GCNConv
import time
import argparse
from torch_geometric.nn import GAE, VGAE, APPNP
from utils import *
import torch.nn.functional as F
def get_args():
    parser = argparse.ArgumentParser(description='CFLP')
    parser.add_argument('--dataset', type=str, default='chameleon')
    parser.add_argument('--datapath', type=str, default='../dataset')
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--name', type=str, default='debug', help='name for this run for logging')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--model_lr', type=float, default=0.01, help='the learning rate of total model')

    parser.add_argument('--val_frac', type=float, default=0.05,help='fraction of edges for validation set')
    parser.add_argument('--test_frac', type=float, default=0.15,help='fraction of edges for testing set')

    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--model', type=str, default='VGAE')

    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--trails', type=int, default=5)
    parser.add_argument('--seed', type=int, default=-1, help='fix random seed if needed')

    args = parser.parse_args()
    args.argv = sys.argv

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    args.device = torch.device('cuda:1' if args.gpu >= -1 else 'cpu')

    return args


class VGNAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGNAEEncoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)

    def forward(self, x, edge_index, args):
        if args.model == 'GAE':
            x = self.linear1(x)
            x = F.normalize(x,p=2,dim=1)  * 1.8
            x = self.propagate(x, edge_index)
            return x

        if args.model == 'VGAE':
            x_ = self.linear1(x)
            x_ = self.propagate(x_, edge_index)

            x = self.linear2(x)
            x = F.normalize(x,p=2,dim=1) * 0.4
            x = self.propagate(x, edge_index)
            return x, x_

        return x


def run_model(args, logger):
    if args.seed > 0:
        set_rng_seed(args.seed)
    # load data
    train_data, val_data, test_data = dataloader(args)
    #model and optim
    model = eval(args.model)(VGNAEEncoder(train_data.x.size(1), 128)).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=0.005)

    step, best_auc, best_ap, best_model = 0, 0, 0, None
    #train
    for epoch in range(args.epoch):
        model.train()
        mi_count, total_cor = 0, 0
        z = model.encode(train_data.x, train_data.edge_index, args)
        loss = model.recon_loss(z, train_data.pos_edge_label_index)


        if args.model == 'VGAE':
            loss = loss + (1 / train_data.edge_index.size(1)) * model.kl_loss()
        optim.zero_grad()
        loss.backward()
        optim.step()


        #evaluation
        model.eval()
        with torch.no_grad():
            z = model.encode(val_data.x, val_data.edge_index, args)
            val_auc, val_ap = model.test(z, val_data.pos_edge_label_index, val_data.neg_edge_label_index)

        if (val_auc+val_ap) > (best_auc+best_ap):
            best_ap = val_ap
            best_auc =val_auc

            model.eval()
            with torch.no_grad():
                z = model.encode(test_data.x, test_data.edge_index, args)
                test_auc, test_ap = model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)


        logger.info('Epoch:{:03d}, train_loss:{:.4f}, val_auc:{:.4f}, val_ap: {:.4f}, text_auc:{:.4f}, text_ap: {:.4f}'.format(epoch, loss, val_auc, val_ap, test_auc, test_ap))

    return test_auc, test_ap


def main(args):
    log_name = f'{args.log_dir}/{args.name}_{args.dataset}_{args.model}_{time.strftime("%Y-%m-%d,%H-%M")}'
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(log_name)
    logger.info(f'args: {args}')

    test_auc_list, test_ap_list = [], []
    for i in range(args.trails):
        test_auc, test_ap = run_model(args, logger)
        test_auc_list.append(test_auc)
        test_ap_list.append(test_ap)
        logger.info('final results:')
        logger.info('====================================================================================')

        logger.info('AUC:{:.4f}+-{:.4f} {}'.format(np.mean(test_auc_list), np.std(test_auc_list), test_auc_list))
        logger.info('AP:{:.4f}+-{:.4f} {}'.format(np.mean(test_ap_list), np.std(test_ap_list), test_ap_list))


if __name__ == "__main__":
    args = get_args()
    main(args)
