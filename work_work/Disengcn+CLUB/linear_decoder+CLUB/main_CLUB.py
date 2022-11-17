import random
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv
from model import Disen_Linear
from model import max_deco
from utils import *
from correlation import *
from torch.nn import ModuleList, Module

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
hyperpm['independence']=1
hyperpm['dropout']=0.4
# hyperpm['device']=device

class Disen_MI(Module):
    def __init__(self, in_dim, hyperpm):
        super(Disen_MI, self).__init__()
        num_dis = int(hyperpm['k'] * (hyperpm['k'] - 1) / 2)
        self.mi_estimators = ModuleList([CLUBSample(hyperpm) for i in range(num_dis)])
        self.gae_model = GAE(encoder=Disen_Linear(in_dim, hyperpm))
        # self.device = hyperpm['device']
    def cor(self, z):
        cor, mi_num=0, 0
        for i in range(hyperpm['k']):
            for j in range(i + 1, hyperpm['k']):
                cor += self.mi_estimators[mi_num](z[:, i * hyperpm['hidden_dim']: (i + 1) * hyperpm['hidden_dim']],
                                               z[:, j * hyperpm['hidden_dim']: (j + 1) * hyperpm['hidden_dim']])
                mi_num += 1
        return cor
    def lld_loss(self, z):
        lld_loss, mi_num = 0, 0
        for i in range(hyperpm['k']):
            for j in range(i + 1, hyperpm['k']):
                lld_loss += self.mi_estimators[mi_num].learning_loss(z[:, i * hyperpm['hidden_dim']: (i + 1) * hyperpm['hidden_dim']],
                                                   z[:, j * hyperpm['hidden_dim']: (j + 1) * hyperpm['hidden_dim']])
                mi_num += 1
        return lld_loss
    def forward(self, x, edge_index):
        z = self.gae_model.encoder(x ,edge_index)
        cor = self.cor(z)
        return z, cor

def train():
    model.train()
    model.mi_estimators.eval()
    z, cor = model.forward(train_data.x, train_data.edge_index)
    gae_loss = model.gae_model.recon_loss(z, train_data.pos_edge_label_index)
    loss = gae_loss + cor
    # print(f'loss:{loss}, cor:{cor}')
    gae_optimizer.zero_grad()
    loss.backward()
    gae_optimizer.step()

    model.mi_estimators.train()
    z, cor = model.forward(train_data.x, train_data.edge_index)
    lld_loss = model.lld_loss(z)
    mi_optimizer.zero_grad()
    lld_loss.backward()
    mi_optimizer.step()

    return loss

@torch.no_grad()
def test(data):
    model.eval()
    z, _ = model.forward(data.x, data.edge_index)
    auc, ap=model.gae_model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    return auc, ap

def run_model():
    step, best_auc, best_ap, best_model, train_loss_list = 0, 0, 0, None, []
    for epoch in range(hyperpm['epoch']):
        loss = train()
        train_loss_list.append(loss)
        val_auc, val_ap = test(val_data)
        print(f'Epoch: {epoch:03d}, train_loss:{loss:.4f}, val_auc: {val_auc:.4f}, val_ap: {val_ap:.4f}')
        if val_auc+val_ap>best_auc+best_ap:
            best_ap = val_ap
            best_auc =val_auc
            torch.save(model.state_dict(), 'best_model_VGAE.pth')
            step = epoch

    model.load_state_dict(torch.load('best_model_VGAE.pth'))
    test_auc, test_ap = test(test_data)
    print(f'the epoch:{step}, test_auc:{test_auc}, test_ap:{test_ap}')
    # visualize(train_loss_list)
    return test_auc, test_ap

for i in range(hyperpm['run_num']):
    hyperpm['seed']=589
    set_rng_seed(hyperpm['seed'])
    print(f'==========================run model {i + 1}==========================')
    log_param(hyperpm)

    train_data, val_data, test_data = dataloader(hyperpm, device)

    model = Disen_MI(in_dim=train_data.x.size(1), hyperpm=hyperpm).to(device)

    mi_params = list(map(id, model.mi_estimators.parameters()))
    rest_params = filter(lambda x: id(x) not in mi_params, model.parameters())
    mi_optimizer = torch.optim.Adam(model.mi_estimators.parameters(), lr=hyperpm['mi_lr'])
    gae_optimizer = torch.optim.Adam(rest_params, lr=hyperpm['model_lr'])

    test_auc, test_ap = run_model()
    total_test_auc, total_test_ap = 0, 0
    total_test_auc = total_test_auc + test_auc
    total_test_ap = total_test_ap + test_ap

    average_test_auc = total_test_auc/(i+1)
    average_test_ap = total_test_ap / (i+1)
    print(f'Model runs {i+1} times, average_test_auc:{average_test_auc}, average_test_ap:{average_test_ap}.')




