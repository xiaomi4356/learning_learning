import random
import torch
import sys
from torch_geometric.nn import GAE, VGAE, GCNConv
from model import Disen_Linear
from utils import *
from correlation import *
from torch.nn import ModuleList
import time
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='DGAE')
    parser.add_argument('--dataset', type=str, default='Synthetic dataset')
    parser.add_argument('--num_c', type=int, default=500, help='node numbers for every community')
    parser.add_argument('--q', type=float, default=0.000000001, help='probability of linking between communities')

    parser.add_argument('--p1', type=float, default=0.16, help='probability of linking within 1st communities')
    parser.add_argument('--p2', type=float, default=0.03, help='probability of linking within 2nd communities')
    parser.add_argument('--p3', type=float, default=0.05, help='probability of linking within 3rd communities')
    parser.add_argument('--p4', type=float, default=0.12, help='probability of linking within 4th communities')
    parser.add_argument('--p5', type=float, default=0.08, help='probability of linking within 4th communities')

    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--name', type=str, default='debug', help='name for this run for logging')

    parser.add_argument('--model', type=str, default='VGAE')
    parser.add_argument('--k', type=int, default=5, help='channels')
    parser.add_argument('--x_dim', type=int, default=16, help='dimension of each channels')
    parser.add_argument('--routit', type=int, default=3, help='iteration of disentangle')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--gpu', type=int, default=-1)


    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--model_lr', type=float, default=0.01, help='the learning rate of total model')
    parser.add_argument('--mi_lr', type=float, default=0.003, help='the learning rate of MI')
    parser.add_argument('--alpha', type=float, default=0.01, help='coefficient of MI term')

    parser.add_argument('--mi_iter', type=int, default=5, help='iteration of MI')
    parser.add_argument('--val_frac', type=float, default=0.05,help='fraction of edges for validation set')
    parser.add_argument('--test_frac', type=float, default=0.1,help='fraction of edges for testing set')
    parser.add_argument('--mi_est', type=str, default='CLUBSample')
    parser.add_argument('--trails', type=int, default=1)
    parser.add_argument('--seed', type=int, default=-1, help='fix random seed if needed')
    parser.add_argument('--verbose', type=int, default=1, help='whether to print per-epoch logs')
    args = parser.parse_args()
    args.argv = sys.argv

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    args.device = torch.device('cuda:1' if args.gpu >= -1 else 'cpu')

    return args


def run_model(args, logger):
    if args.seed > 0:
        set_rng_seed(args.seed)
    # load data
    train_data, val_data, test_data, data = dataloader(args)
    #model and optim
    model = eval(args.model)(encoder=Disen_Linear(train_data.x.size(1), args)).to(args.device)
    optim = torch.optim.Adam(model.parameters(), args.model_lr)
    num_dis = int(args.k*(args.k - 1) / 2)
    mi_est = ModuleList([eval(args.mi_est)(args.x_dim, args.x_dim, args.x_dim) for i in range(num_dis)]).to(args.device)
    mi_optim = torch.optim.Adam(mi_est.parameters(), lr=args.mi_lr)

    step, best_auc, best_ap, best_model = 0, 0, 0, None
    #train
    for epoch in range(args.epoch):
        model.train()
        mi_count, total_cor = 0, 0
        mi_est.eval()
        z = model.encode(train_data.x, train_data.edge_index, args)

        for i in range(args.k):
            for j in range(i + 1, args.k):
                cor = mi_est[mi_count](z[:, i * args.x_dim: (i + 1) * args.x_dim],
                                       z[:, j * args.x_dim: (j + 1) * args.x_dim])
                total_cor = total_cor + cor
                mi_count += 1

        recon_loss = model.recon_loss(z, train_data.pos_edge_label_index)

        loss = recon_loss + args.alpha * total_cor
        if args.model == 'VGAE':
            # loss = loss+(1 / train_data.num_nodes)*model.kl_loss()
            loss = loss + (1 / train_data.edge_index.size(1)) * model.kl_loss()
        optim.zero_grad()
        loss.backward()
        optim.step()

        for k in range(args.mi_iter):
            model.eval()
            mi_est.train()
            z = model.encode(train_data.x, train_data.edge_index, args)
            total_lld_loss, mi_count = 0, 0
            for i in range(args.k):
                for j in range(i + 1, args.k):
                    lld_loss = mi_est[mi_count].learning_loss(z[:, i * args.x_dim: (i + 1) * args.x_dim],
                                                              z[:, j * args.x_dim: (j + 1) * args.x_dim])
                    total_lld_loss = total_lld_loss + lld_loss
                    mi_count += 1

            mean_loss = total_lld_loss / (mi_count + 1)
            mi_optim.zero_grad()
            total_lld_loss.backward()
            mi_optim.step()
        if epoch%10==0:
            torch.save(model.state_dict(), '/home/dell/zxj/feat/main_5/'+f'syn_data_epoch_{epoch}.pth')
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

            if args.verbose:
                logger.info('Epoch:{:03d}, train_loss:{:.4f}, val_auc:{:.4f}, val_ap: {:.4f}, text_auc:{:.4f}, text_ap: {:.4f}'.format(epoch, loss, val_auc, val_ap, test_auc, test_ap))

        else:
            if args.verbose:
                #logger.info('Epoch:{:03d}, train_loss:{:.4f}, val_auc:{:.4f}, val_ap: {:.4f}, total_cor:{:.4f}, total_lld_loss:{:.4f}'.format(epoch, loss, val_auc, val_ap, total_cor, total_lld_loss))
                logger.info('Epoch:{:03d}, train_loss:{:.4f}, val_auc:{:.4f}, val_ap: {:.4f}'.format(epoch, loss, val_auc, val_ap))

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
