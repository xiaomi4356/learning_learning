import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.nn import GAE, VGAE, GCNConv
from model import Disen_Linear
from utils import *
from correlation import *
from torch.nn import ModuleList

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
hyperpm=dict()
hyperpm['dataset']='Citeseer'
hyperpm['epoch']=300
hyperpm['model_lr']=0.01
hyperpm['mi_lr']=0.005
hyperpm['alpha']=0.01
# hyperpm['k']=7
hyperpm['x_dim']=64
hyperpm['run_num']=1
hyperpm['mi_iter']=5
hyperpm['routit']=3
hyperpm['dropout']=0
hyperpm['model']='VGAE'
hyperpm['mi_estimators']='CLUBSample'

#面临的问题是两个模型，两个优化器
def train():
    model.train()
    mi_count, total_cor = 0, 0
    mi_estimators.eval()
    z = model.encode(train_data.x, train_data.edge_index, hyperpm)

    for i in range(hyperpm['k']):
        for j in range(i + 1, hyperpm['k']):
            cor = mi_estimators[mi_count](z[:, i * hyperpm['x_dim']: (i + 1) * hyperpm['x_dim']],
                                                  z[:, j * hyperpm['x_dim']: (j + 1) * hyperpm['x_dim']])
            total_cor = total_cor+cor
            mi_count += 1

    recon_loss = model.recon_loss(z, train_data.pos_edge_label_index)

    loss = recon_loss + hyperpm['alpha']*total_cor
    if hyperpm['model']=='VGAE':
        # loss = loss+(1 / train_data.num_nodes)*model.kl_loss()
        loss = loss+(1 / train_data.edge_index.size(1))*model.kl_loss()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for k in range(hyperpm['mi_iter']):
        model.eval()
        mi_estimators.train()
        z = model.encode(train_data.x, train_data.edge_index, hyperpm)
        total_lld_loss, mi_count = 0, 0
        for i in range(hyperpm['k']):
            for j in range(i + 1, hyperpm['k']):
                lld_loss = mi_estimators[mi_count].learning_loss(z[:, i * hyperpm['x_dim']: (i + 1) * hyperpm['x_dim']],
                                                   z[:, j * hyperpm['x_dim']: (j + 1) * hyperpm['x_dim']])
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
    z = model.encode(data.x, data.edge_index, hyperpm)
    auc, ap=model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    return auc, ap

def run_model():
    step, best_auc, best_ap, best_model = 0, 0, 0, None,

    for epoch in range(hyperpm['epoch']):
        loss, total_cor, total_lld_loss = train()

        val_auc, val_ap = test(val_data)
        if epoch%50==0:
            print(f'Epoch: {epoch:03d}, train_loss:{loss:.6f}, val_auc: {val_auc:.6f}, val_ap: {val_ap:.6f}, total_cor:{total_cor:.6f}, total_lld_loss:{total_lld_loss:.6f}')
        if (val_auc+val_ap) > (best_auc+best_ap):
            best_ap = val_ap
            best_auc =val_auc
            torch.save(model.state_dict(), 'hyp_sens.pth')
            step = epoch

    model.load_state_dict(torch.load('hyp_sens.pth'))

    test_auc, test_ap = test(test_data)
    test_auc_list.append(test_auc)
    test_ap_list.append(test_ap)
    print(f'the epoch:{step}, test_auc:{test_auc:.6f}, test_ap:{test_ap:.6f}')

    return test_auc, test_ap, test_auc_list, test_ap_list


test_auc_list, test_ap_list = [], []

for hyperpm['k'] in range(2, 20):
    for i in range(hyperpm['run_num']):
        hyperpm['seed'] = 4839
        set_rng_seed(hyperpm['seed'])
        print(f'==========================run model {i + 1}==========================')
        # log_param(hyperpm)

        train_data, val_data, test_data = dataloader(hyperpm, device)
        print(train_data)
        model = eval(hyperpm['model'])(encoder=Disen_Linear(train_data.x.size(1), hyperpm)).to(device)

        optimizer = torch.optim.Adam(model.parameters(), hyperpm['model_lr'])

        num_dis = int(hyperpm['k'] * (hyperpm['k'] - 1) / 2)
        mi_estimators = ModuleList([eval(hyperpm['mi_estimators'])(hyperpm['x_dim'], hyperpm['x_dim'], hyperpm['x_dim']) for i in range(num_dis)]).to(device)
        mi_optimizer = torch.optim.Adam(mi_estimators.parameters(), lr=hyperpm['mi_lr'])

        test_auc, test_ap, test_auc_list, test_ap_list= run_model()

        log_param(hyperpm)
        print(f'Model runs {i+1} times, average_test_auc:{test_auc:.6f}, average_test_ap:{test_ap:.6f}.')
        print(f'Citeseer_channel={test_auc_list}')
        print(f'Citeseer_channel={test_ap_list}')
visualize(test_auc_list)
visualize(test_ap_list)



