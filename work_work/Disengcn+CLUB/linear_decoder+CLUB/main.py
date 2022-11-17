import random
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv
from model import Disen_Linear
from model import max_deco
from utils import *
from correlation import *
from torch.nn import ModuleList

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
hyperpm=dict()
hyperpm['dataset']='Cora'
hyperpm['epoch']=350
hyperpm['model_lr']=0.008
hyperpm['mi_lr']=0.01
hyperpm['k']=4
hyperpm['out_dim']=256
hyperpm['hidden_dim']=64
hyperpm['x_dim']=64
hyperpm['y_dim']=64
hyperpm['run_num']=1
hyperpm['routit']=1
hyperpm['dropout']=0.4
# hyperpm['device']=device
#面临的问题是两个模型，两个优化器
def train():
    gae_model.train()
    cor, mi_count, lld_loss, total_cor = 0, 0, 0, 0
    mi_estimators.eval()

    z = gae_model.encode(train_data.x, train_data.edge_index)
    for i in range(hyperpm['k']):
        for j in range(i + 1, hyperpm['k']):
            cor = mi_estimators[mi_count](z[:, i * hyperpm['hidden_dim']: (i + 1) * hyperpm['hidden_dim']],
                                                  z[:, j * hyperpm['hidden_dim']: (j + 1) * hyperpm['hidden_dim']])
            total_cor = total_cor+cor
            # print(f'i:{i+1}, j:{j+1}, cor:{cor}')
            mi_count += 1

    gae_loss = gae_model.recon_loss(z, train_data.pos_edge_label_index)
    loss = gae_loss + total_cor
    gae_optimizer.zero_grad()
    loss.backward()
    gae_optimizer.step()
    mi_count, total_lld_loss = 0,0
    mi_estimators.train()
    gae_model.eval()
    z = gae_model.encode(train_data.x, train_data.edge_index)
    for i in range(hyperpm['k']):
        for j in range(i + 1, hyperpm['k']):
            lld_loss = mi_estimators[mi_count].learning_loss(z[:, i * hyperpm['hidden_dim']: (i + 1) * hyperpm['hidden_dim']],
                                               z[:, j * hyperpm['hidden_dim']: (j + 1) * hyperpm['hidden_dim']])
            total_lld_loss = total_lld_loss+lld_loss
            # print(f'i:{i+1}, j:{j+1}, lld_loss:{lld_loss}')
            mi_count += 1
    # print(f'total_lld_loss:{total_lld_loss}')
    mean_loss = total_lld_loss/(mi_count+1)
    mi_optimizer.zero_grad()
    mean_loss.backward()
    mi_optimizer.step()

    return float(loss), total_cor, total_lld_loss

@torch.no_grad()
def test(data):
    gae_model.eval()
    z = gae_model.encode(data.x, data.edge_index)
    auc, ap=gae_model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    return auc, ap

def run_model():
    step, best_auc, best_ap, best_model = 0, 0, 0, None,
    train_loss_list, total_cor_list, total_lld_list = [], [], []
    for epoch in range(hyperpm['epoch']):
        # print(f'epoch:{epoch}')
        loss, total_cor, total_lld_loss = train()
        train_loss_list.append(loss)
        total_cor_list.append(total_cor)
        total_lld_list.append(total_lld_loss)

        val_auc, val_ap = test(val_data)
        print(f'Epoch: {epoch:03d}, train_loss:{loss:.4f}, val_auc: {val_auc:.4f}, val_ap: {val_ap:.4f}, total_cor:{total_cor}, total_lld_loss:{total_lld_loss}')
        if val_auc+val_ap >best_auc+best_ap:
            best_ap = val_ap
            best_auc =val_auc
            torch.save(gae_model.state_dict(), 'best_model_VGAE.pth')
            step = epoch

    gae_model.load_state_dict(torch.load('best_model_VGAE.pth'))
    test_auc, test_ap = test(test_data)
    print(f'the epoch:{step}, test_auc:{test_auc}, test_ap:{test_ap}')
    # visualize(train_loss_list)
    # visualize(total_cor_list)
    # visualize(total_lld_list)
    return test_auc, test_ap

total_test_auc, total_test_ap = 0, 0
for i in range(hyperpm['run_num']):
    hyperpm['seed']=589
    set_rng_seed(hyperpm['seed'])
    print(f'==========================run model {i + 1}==========================')
    log_param(hyperpm)

    train_data, val_data, test_data = dataloader(hyperpm, device)

    gae_model = GAE(encoder=Disen_Linear(train_data.x.size(1), hyperpm)).to(device)
    gae_optimizer = torch.optim.Adam(gae_model.parameters(), hyperpm['model_lr'])

    num_dis = int(hyperpm['k'] * (hyperpm['k'] - 1) / 2)
    mi_estimators = ModuleList([CLUBSample(hyperpm) for i in range(num_dis)]).to(device)
    mi_optimizer = torch.optim.Adam(mi_estimators.parameters(), lr=hyperpm['mi_lr'])

    test_auc, test_ap = run_model()
    total_test_auc = total_test_auc + test_auc
    total_test_ap = total_test_ap + test_ap

    average_test_auc = total_test_auc/(i+1)
    average_test_ap = total_test_ap / (i+1)
    print(f'Model runs {i+1} times, average_test_auc:{average_test_auc}, average_test_ap:{average_test_ap}.')

# model.load_state_dict(torch.load('best_model_VGAE.pth'))
# z=model.encode(test_data.x, test_data.edge_index)
# visualize_dimreduc(z, test_data.y)
