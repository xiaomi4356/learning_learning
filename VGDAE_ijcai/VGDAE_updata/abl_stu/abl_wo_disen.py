import random
import torch
from torch_geometric.nn import GAE, VGAE, GCNConv
from torch.nn import Linear, Module
from utils import *

hyperpm=dict()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hyperpm['dataset']='Citeseer'
hyperpm['epoch']=300
hyperpm['lr']=0.004
hyperpm['k']=4
hyperpm['x_dim']=64
hyperpm['run_num']=10
hyperpm['model']='VGAE'

class Encoder(Module):
    def __init__(self, input, hyperpm):
        super(Encoder, self).__init__()
        self.linear = Linear(in_features=input, out_features=hyperpm['k']*hyperpm['x_dim'])
        if hyperpm['model'] == 'VGAE':
            self.linear_ = Linear(in_features=input, out_features=hyperpm['k']*hyperpm['x_dim'])
    def forward(self, x):
        if hyperpm['model']=='GAE':
            x = self.linear(x)
            return x

        if hyperpm['model'] == 'VGAE':
            mu = self.linear(x)
            log = self.linear_(x)
            return mu, log
def train():
    model.train()
    z = model.encode(train_data.x)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    if hyperpm['model']=='VGAE':
        loss = loss + (1 / train_data.edge_index.size(1)) * model.kl_loss()
        # loss = loss + (1 / train_data.num_nodes) * model.kl_loss()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x)
    auc, ap=model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    return auc, ap
def run_model():
    loss, step, best_auc, best_ap, best_model, train_loss_list = 0, 0, 0, 0, None, []
    for epoch in range(hyperpm['epoch']):
        loss = train()

        val_auc, val_ap = test(val_data)

        print(f'Epoch: {epoch:03d}, train_loss:{loss:.4f}, val_auc: {val_auc:.4f}, val_ap: {val_ap:.4f}')
        if (val_auc+val_ap) > (best_auc+best_ap):
            best_ap = val_ap
            best_auc =val_auc
            torch.save(model.state_dict(), 'abl_stu_wo_club.pth')
            step = epoch

    model.load_state_dict(torch.load('abl_stu_wo_club.pth'))
    test_auc, test_ap = test(test_data)
    print(f'the epoch:{step}, test_auc:{test_auc}, test_ap:{test_ap}')
    return test_auc, test_ap
total_test_auc, total_test_ap = 0, 0
for i in range(hyperpm['run_num']):
    hyperpm['seed']=random.randint(1, 10000)
    set_rng_seed(hyperpm['seed'])
    print(f'==========================run model {i + 1}==========================')
    # log_param(hyperpm)

    train_data, val_data, test_data = dataloader(hyperpm, device)

    model = eval(hyperpm['model'])(encoder=Encoder(train_data.x.size(1), hyperpm)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperpm['lr'])
    print(model)
    # writer.add_graph(model)
    test_auc, test_ap = run_model()

    total_test_auc = total_test_auc + test_auc
    total_test_ap = total_test_ap + test_ap

    average_test_auc = total_test_auc/(i+1)
    average_test_ap = total_test_ap / (i+1)
    log_param(hyperpm)
    print(f'Model runs {i+1} times, average_test_auc:{average_test_auc}, average_test_ap:{average_test_ap}.')