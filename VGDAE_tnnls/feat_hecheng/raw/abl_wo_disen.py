import random
import torch
from torch_geometric.nn import GAE, VGAE, GCNConv
from torch.nn import Linear, Module
from utils import *
import argparse
import sys
import time
def get_args():
    parser = argparse.ArgumentParser(description='DGAE')
    parser.add_argument('--dataset', type=str, default='Synthetic dataset')
    parser.add_argument('--num_c', type=int, default=500, help='node numbers for every community')
    parser.add_argument('--q', type=float, default=0.000001, help='probability of linking between communities')

    parser.add_argument('--p1', type=float, default=0.01, help='probability of linking within 1st communities')
    parser.add_argument('--p2', type=float, default=0.2, help='probability of linking within 2nd communities')
    parser.add_argument('--p3', type=float, default=0.05, help='probability of linking within 3rd communities')
    parser.add_argument('--p4', type=float, default=0.4, help='probability of linking within 4th communities')
    parser.add_argument('--p5', type=float, default=0.08, help='probability of linking within 4th communities')


    parser.add_argument('--k', type=int, default=5, help='channels')
    parser.add_argument('--x_dim', type=int, default=8, help='dimension of each channels')

    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--name', type=str, default='debug', help='name for this run for logging')

    parser.add_argument('--model', type=str, default='VGAE')
    parser.add_argument('--model_lr', type=float, default=0.01)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--out_dim', type=int, default=40)
    parser.add_argument('--trails', type=int, default=1)

    parser.add_argument('--val_frac', type=float, default=0.05, help='fraction of edges for validation set')
    parser.add_argument('--test_frac', type=float, default=0.1, help='fraction of edges for testing set')

    args = parser.parse_args()
    args.argv = sys.argv

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    args.device = torch.device('cuda:0' if args.gpu >= -1 else 'cpu')

    return args


class Encoder(Module):
    def __init__(self, input, args):
        super(Encoder, self).__init__()
        self.linear = Linear(in_features=input, out_features=args.k*args.x_dim)
        if args.model == 'VGAE':
            self.linear_ = Linear(in_features=input, out_features=args.k*args.x_dim)
    def forward(self, x, args):
        if args.model =='GAE':
            x = self.linear(x)
            return x

        if args.model == 'VGAE':
            mu = self.linear(x)
            log = self.linear_(x)
            return mu, log


def run_model(args, logger):

    # load data
    train_data, val_data, test_data, data = dataloader(args)

    #model and optim

    model = eval(args.model)(encoder=Encoder(train_data.x.size(1),args)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)

    step, best_auc, best_ap, best_model = 0, 0, 0, None
    #train
    for epoch in range(args.epoch):
        model.train()
        z = model.encode(train_data.x, args)
        loss = model.recon_loss(z, train_data.pos_edge_label_index)
        if args.model == 'VGAE':
            loss = loss + (1 / train_data.edge_index.size(1)) * model.kl_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #evaluation
        model.eval()
        with torch.no_grad():
            z = model.encode(val_data.x, args)
            val_auc, val_ap = model.test(z, val_data.pos_edge_label_index, val_data.neg_edge_label_index)

        if (val_auc+val_ap) > (best_auc+best_ap):
            best_ap = val_ap
            best_auc =val_auc

            model.eval()
            with torch.no_grad():
                z = model.encode(test_data.x, args)
                test_auc, test_ap = model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
                torch.save(model.state_dict(), 'syn_raw.pth')

        logger.info('Epoch:{:03d}, train_loss:{:.4f}, val_auc:{:.4f}, val_ap: {:.4f}, text_auc:{:.4f}, text_ap: {:.4f}'.format(epoch, loss, val_auc, val_ap, test_auc, test_ap))
        logger.info('Epoch:{:03d}, train_loss:{:.4f}, val_auc:{:.4f}, val_ap: {:.4f}'.format(epoch, loss, val_auc, val_ap))
    return test_auc, test_ap

def main(args):
    log_name = f'{args.log_dir}/{args.name}_{args.dataset}_{args.model}_{time.strftime("%Y-%m-%d,%H:%M")}'
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
