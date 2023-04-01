import random
import torch
from torch_geometric.nn import GAE, VGAE, GCNConv
from model import Disen_Linear
from model import max_deco
from utils import *
from correlation import *
from torch.nn import ModuleList
import argparse
import hyperopt

seed =  78
set_rng_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='cora')
parser.add_argument('--epoch', type=int, default=300,  help='Max number of epochs to train.')
parser.add_argument('--model_lr', type=float, default=0.01, help='model learning rate.')
parser.add_argument('--mi_lr', type=float, default=0.003, help='MI learning rate.')
parser.add_argument('--alpha', type=float, default=0.01, help='MI regularization coefficient.')
parser.add_argument('--k', type=int, default=5,help='Number of channels.')
parser.add_argument('--hid_dim', type=int, default=16, help='Number of hidden units per channels.')
parser.add_argument('--mi_iter', type=int, default=5, help='MI iterations.')
parser.add_argument('--routit', type=int, default=6, help='Number of iterations when routing.')
parser.add_argument('--model', type=str, default='VGAE')
parser.add_argument('--mi_est', type=str, default='CLUBSample')
parser.add_argument('--runs', type=int, default=1,  help='model runs time.')

args = parser.parse_args()



#面临的问题是两个模型，两个优化器
def train():
    model.train()
    mi_count, total_cor = 0, 0
    mi_estimators.eval()
    z = model.encode(train_data.x, train_data.edge_index, args.model)

    for i in range(args.k):
        for j in range(i + 1, args.k):
            cor = mi_estimators[mi_count](z[:, i * args.hid_dim: (i + 1) * args.hid_dim],
                                                  z[:, j * args.hid_dim: (j + 1) * args.hid_dim])
            total_cor = total_cor+cor
            mi_count += 1

    recon_loss = model.recon_loss(z, train_data.pos_edge_label_index)

    loss = recon_loss + args.alpha*total_cor
    if args.model=='VGAE':
        # loss = loss+(1 / train_data.num_nodes)*model.kl_loss()
        loss = loss+(1 / train_data.edge_index.size(1))*model.kl_loss()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for k in range(args.mi_iter):
        model.eval()
        mi_estimators.train()
        z = model.encode(train_data.x, train_data.edge_index, args.model)
        total_lld_loss, mi_count = 0, 0
        for i in range(args.k):
            for j in range(i + 1, args.k):
                lld_loss = mi_estimators[mi_count].learning_loss(z[:, i *args.hid_dim: (i + 1) * args.hid_dim],
                                                   z[:, j * args.hid_dim: (j + 1) * args.hid_dim])
                total_lld_loss = total_lld_loss+lld_loss
                mi_count += 1

        mean_loss = total_lld_loss/(mi_count+1)
        mi_optimizer.zero_grad()
        total_lld_loss.backward()
        mi_optimizer.step()

    return loss, total_cor, total_lld_loss

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index, args.model)
    auc, ap=model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    return auc, ap

def run_model():
    step, best_auc, best_ap, best_model = 0, 0, 0, None,

    for epoch in range(args.epoch):
        loss, total_cor, total_lld_loss = train()

        val_auc, val_ap = test(val_data)

        print(f'Epoch: {epoch:03d}, train_loss:{loss:.6f}, val_auc: {val_auc:.6f}, val_ap: {val_ap:.6f}, total_cor:{total_cor:.6f}, total_lld_loss:{total_lld_loss:.6f}')
        if (val_auc+val_ap) > (best_auc+best_ap):
            best_ap = val_ap
            best_auc =val_auc
            torch.save(model.state_dict(), 'cora.pth')
            step = epoch

    return best_auc, best_ap


train_data, val_data, test_data = dataloader(args.dataname, device)
model = eval(args.model)(encoder=Disen_Linear(train_data.x.size(1), args.k, args.hid_dim, args.model, args.routit)).to(device)
optimizer = torch.optim.Adam(model.parameters(), args.model_lr)

num_dis = int(args.k* (args.k - 1) / 2)
mi_estimators = ModuleList([eval(args.mi_est)(args.hid_dim, args.hid_dim, args.hid_dim) for i in range(num_dis)]).to(device)
mi_optimizer = torch.optim.Adam(mi_estimators.parameters(), lr=args.mi_lr)


def f_auc():
    auc, ap = run_model()
    return -auc

space = {'model_lr': hyperopt.hp.loguniform('model_lr', -8, 0),
         'mi_lr': hyperopt.hp.loguniform('mi_lr', -8, 0),
         'alpha':hyperopt.hp.loguniform('alpha', -8, 0),
         'k': hyperopt.hp.quniform('k', 2, 10, 1),
         'dropout': hyperopt.hp.uniform('dropout', 0, 0.9),
         'routit': hyperopt.hp.quniform('routit', 2, 8, 1),
         'mi_iter':hyperopt.hp.quniform('mi_iter', 2, 8, 1),
         }


trials = hyperopt.Trials()
best = hyperopt.fmin(f_auc, space, algo=hyperopt.tpe.suggest, max_evals=1000, trials=trials)
print('best:')
print(best)