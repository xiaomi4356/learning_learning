
import random
import torch
from torch_geometric.nn import GCNConv, ARGVA, ARGA
import torch
import torch.nn.functional as F
from utils import *

hyperpm=dict()
device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
hyperpm['dataset']='Citeseer'
hyperpm['epoch']=3000
hyperpm['run_num']=10

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
def train():
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
            torch.save(model.state_dict(), 'arvga_cora_bias.pth')
            step = epoch

    model.load_state_dict(torch.load('arvga_cora_bias.pth'))
    test_auc, test_ap = test(test_data)
    print(f'the epoch:{step}, test_auc:{test_auc}, test_ap:{test_ap}')
    return test_auc, test_ap

total_test_auc, total_test_ap = 0, 0
for i in range(hyperpm['run_num']):
    hyperpm['seed'] = random.randint(1, 10000)
    set_rng_seed(hyperpm['seed'])
    print(f'==========================run model {i + 1}==========================')
    # log_param(hyperpm)

    train_data, val_data, test_data = dataloader(hyperpm, device)
    encoder = Encoder(train_data.x.size(1), hidden_channels=64, out_channels=32)
    discriminator = Discriminator(in_channels=32, hidden_channels1=16, hidden_channels2=64,
                                  out_channels=128)
    model = ARGVA(encoder, discriminator).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    print(model)
    test_auc, test_ap = run_model()
    total_test_auc = total_test_auc + test_auc
    total_test_ap = total_test_ap + test_ap

    average_test_auc = total_test_auc/(i+1)
    average_test_ap = total_test_ap / (i+1)
    log_param(hyperpm)
    print(f'Model runs {i+1} times, average_test_auc:{average_test_auc}, average_test_ap:{average_test_ap}.')