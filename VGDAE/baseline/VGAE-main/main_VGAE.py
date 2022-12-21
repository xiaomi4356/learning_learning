import random
import torch
import torch.nn as nn
from torch_geometric.nn import GAE, VGAE, GCNConv
from utils import *
import torch.nn.functional as F

hyperpm=dict()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hyperpm['dataset']='Pubmed'
hyperpm['epoch']=200
hyperpm['run_num']=10
hyperpm['model']='VGAE'

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

def train():
    model.train()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    if hyperpm['model']=='VGAE':
        # loss = loss + (1 / train_data.edge_index.size(1)) * model.kl_loss()
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
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
            torch.save(model.state_dict(), 'gae_cora_bias.pth')
            step = epoch

    model.load_state_dict(torch.load('gae_cora_bias.pth'))
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
    if hyperpm['model'] == 'GAE':
        model = eval(hyperpm['model'])(GCNEncoder(train_data.x.size(1), 16)).to(device)
    if hyperpm['model'] == 'VGAE':
        model = eval(hyperpm['model'])(VariationalGCNEncoder(train_data.x.size(1), 16)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print(model)
    # writer.add_graph(model)
    test_auc, test_ap = run_model()
    total_test_auc = total_test_auc + test_auc
    total_test_ap = total_test_ap + test_ap

    average_test_auc = total_test_auc/(i+1)
    average_test_ap = total_test_ap / (i+1)
    log_param(hyperpm)
    print(f'Model runs {i+1} times, average_test_auc:{average_test_auc}, average_test_ap:{average_test_ap}.')

