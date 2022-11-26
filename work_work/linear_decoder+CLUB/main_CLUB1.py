import random
import torch
from torch_geometric.nn import GAE, VGAE, GCNConv
from model import Disen_Linear
from model import max_deco
from utils import *
from correlation import *
from torch.nn import ModuleList

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
hyperpm=dict()
hyperpm['dataset']='Pubmed'
hyperpm['epoch']=200
hyperpm['model_lr']=0.001
hyperpm['mi_lr']=0.003
hyperpm['alpha']=1
hyperpm['k']=4
hyperpm['out_dim']=256
hyperpm['hidden_dim']=64
hyperpm['x_dim']=64
hyperpm['y_dim']=64
hyperpm['run_num']=20
hyperpm['mi_iter']=3
hyperpm['routit']=1
hyperpm['dropout']=0
hyperpm['model']='VGAE'

#面临的问题是两个模型，两个优化器
def train():
    model.train()
    mi_count, total_cor = 0, 0
    mi_estimators.eval()
    z = model.encode(train_data.x, train_data.edge_index, hyperpm)

    for i in range(hyperpm['k']):
        for j in range(i + 1, hyperpm['k']):
            cor = mi_estimators[mi_count](z[:, i * hyperpm['hidden_dim']: (i + 1) * hyperpm['hidden_dim']],
                                                  z[:, j * hyperpm['hidden_dim']: (j + 1) * hyperpm['hidden_dim']])
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
                lld_loss = mi_estimators[mi_count].learning_loss(z[:, i * hyperpm['hidden_dim']: (i + 1) * hyperpm['hidden_dim']],
                                                   z[:, j * hyperpm['hidden_dim']: (j + 1) * hyperpm['hidden_dim']])
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
    train_loss_list, total_cor_list, total_lld_list = [], [], []
    for epoch in range(hyperpm['epoch']):
        loss, total_cor, total_lld_loss = train()
        train_loss_list.append(loss)
        total_cor_list.append(total_cor)
        total_lld_list.append(total_lld_loss)

        val_auc, val_ap = test(val_data)
        print(f'Epoch: {epoch:03d}, train_loss:{loss:.6f}, val_auc: {val_auc:.6f}, val_ap: {val_ap:.6f}, total_cor:{total_cor:.6f}, total_lld_loss:{total_lld_loss:.6f}')
        if (val_auc+val_ap) > (best_auc+best_ap):
            best_ap = val_ap
            best_auc =val_auc
            torch.save(model.state_dict(), 'club_vgae_pubmed.pth')
            step = epoch

    model.load_state_dict(torch.load('club_vgae_pubmed.pth'))

    test_auc, test_ap = test(test_data)
    print(f'the epoch:{step}, test_auc:{test_auc:.6f}, test_ap:{test_ap:.6f}')
    # visualize(train_loss_list)
    # visualize(total_cor_list)
    # visualize(total_lld_list)
    return test_auc, test_ap
test_auc_list, test_ap_list = [], []
total_test_auc, total_test_ap = 0, 0
for i in range(hyperpm['run_num']):
    hyperpm['seed']=random.randint(1, 10000)
    set_rng_seed(hyperpm['seed'])
    print(f'==========================run model {i + 1}==========================')
    # log_param(hyperpm)

    train_data, val_data, test_data = dataloader(hyperpm, device)
    model = eval(hyperpm['model'])(encoder=Disen_Linear(train_data.x.size(1), hyperpm)).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), hyperpm['model_lr'])

    num_dis = int(hyperpm['k'] * (hyperpm['k'] - 1) / 2)
    mi_estimators = ModuleList([CLUBSample(hyperpm['x_dim'], hyperpm['y_dim'], hyperpm['hidden_dim']) for i in range(num_dis)]).to(device)
    mi_optimizer = torch.optim.Adam(mi_estimators.parameters(), lr=hyperpm['mi_lr'])

    test_auc, test_ap = run_model()
    test_auc_list.append(test_auc)
    test_ap_list.append(test_ap)
    total_test_auc = total_test_auc + test_auc
    total_test_ap = total_test_ap + test_ap

    average_test_auc = total_test_auc/(i+1)
    average_test_ap = total_test_ap / (i+1)
    log_param(hyperpm)
    print(f'Model runs {i+1} times, average_test_auc:{average_test_auc:.6f}, average_test_ap:{average_test_ap:.6f}.')

# model.load_state_dict(torch.load('best_model_VGAE.pth'))
# z=model.encode(test_data.x, test_data.edge_index)
# visualize_dimreduc(z, test_data.y)
# print(f'test_auc_list：{test_auc_list}')
# print(f'test_ap_list:{test_ap_list}')
# visualize(test_auc_list), visualize(test_ap_list)
