import random
import torch
from torch_geometric.nn import GCNConv, ARGVA, ARGA
import torch
import torch.nn.functional as F
from utils import *
import sys
import time
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='CFLP')
    parser.add_argument('--dataset', type=str, default='squirrel')
    parser.add_argument('--datapath', type=str, default='../dataset')
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--name', type=str, default='debug', help='name for this run for logging')
    parser.add_argument('--epoch', type=int, default=200)

    parser.add_argument('--val_frac', type=float, default=0.05,help='fraction of edges for validation set')
    parser.add_argument('--test_frac', type=float, default=0.15,help='fraction of edges for testing set')

    parser.add_argument('--model', type=str, default='VGAE')
    parser.add_argument('--mi_est', type=str, default='CLUBSample')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--trails', type=int, default=5)
    parser.add_argument('--seed', type=int, default=-1, help='fix random seed if needed')

    args = parser.parse_args()
    args.argv = sys.argv

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    args.device = torch.device('cuda:1' if args.gpu >= -1 else 'cpu')

    return args

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1,hidden_channels2, out_channels):
        super(Discriminator, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels1)
        self.lin2 = torch.nn.Linear(hidden_channels1, hidden_channels2)
        self.lin3 = torch.nn.Linear(hidden_channels2, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return x


def run_model(args, logger):

    # load data
    train_data, val_data, test_data = dataloader(args)

    encoder = Encoder(train_data.x.size(1), hidden_channels=64, out_channels=32)
    discriminator = Discriminator(in_channels=32, hidden_channels1=16, hidden_channels2=64,
                                  out_channels=128)
    model = ARGVA(encoder, discriminator).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)


    step, best_auc, best_ap, best_model = 0, 0, 0, None
    #train
    for epoch in range(args.epoch):
        model.train()
        z = model.encode(train_data.x, train_data.edge_index)

        for i in range(5):
            discriminator_optimizer.zero_grad()
            discriminator_loss = model.discriminator_loss(z)
            discriminator_loss.backward()
            discriminator_optimizer.step()

        loss = model.recon_loss(z, train_data.pos_edge_label_index)
        loss = loss + model.reg_loss(z)
        loss = loss + (1 / train_data.edge_index.size(1)) * model.kl_loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #evaluation
        model.eval()
        with torch.no_grad():
            z = model.encode(val_data.x, val_data.edge_index)
            val_auc, val_ap = model.test(z, val_data.pos_edge_label_index, val_data.neg_edge_label_index)

        if (val_auc+val_ap) > (best_auc+best_ap):
            best_ap = val_ap
            best_auc =val_auc

            model.eval()
            with torch.no_grad():
                z = model.encode(test_data.x, test_data.edge_index)
                test_auc, test_ap = model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
                # torch.save(model.state_dict(), 'syn_GAE.pth')

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

