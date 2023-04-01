import random
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv
from encoder import Disen_Linear
from decoder import max_deco
from utils import *
from DistanceCorrelation import *
hyperpm=dict()
hyperpm['dataset']='Cora'
hyperpm['epoch']=300
hyperpm['lr']=0.008
hyperpm['k']=4
hyperpm['out_dim']=256
hyperpm['hidden_dim']=64
hyperpm['run_num']=10
hyperpm['routit']=1
hyperpm['dropout']=0.4
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    cor_loss=0
    for i in range(hyperpm['k']):
        for j in range(i + 1, hyperpm['k']):
            cor_loss += DistanceCorrelation(z[:, i * hyperpm['hidden_dim']: (i + 1) * hyperpm['hidden_dim']],
                                       z[:, j * hyperpm['hidden_dim']: (j + 1) * hyperpm['hidden_dim']])

    loss = model.recon_loss(z, train_data.pos_edge_label_index)+cor_loss/4
    # print(f'cor_loss:{cor_loss}, cor_loss/4:{cor_loss/4}')
    # loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    auc, ap=model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    return auc, ap

def run_model():
    step, best_auc, best_ap, best_model, train_loss_list = 0, 0, 0, None, []
    for epoch in range(hyperpm['epoch']):
        loss = train()
        train_loss_list.append(loss)
        val_auc, val_ap = test(val_data)
        print(f'Epoch: {epoch:03d}, train_loss:{loss:.4f}, val_auc: {val_auc:.4f}, val_ap: {val_ap:.4f}')
        if val_auc>best_auc:
            best_ap = val_ap
            best_auc =val_auc
            torch.save(model.state_dict(), 'best_model_VGAE.pth')
            step = epoch

    model.load_state_dict(torch.load('best_model_VGAE.pth'))
    test_auc, test_ap = test(test_data)
    print(f'the epoch:{step}, test_auc:{test_auc}, test_ap:{test_ap}')
    # visualize(train_loss_list)
    return test_auc, test_ap

total_test_auc, total_test_ap = 0, 0
for i in range(hyperpm['run_num']):
    hyperpm['seed']=random.randint(1,1000)
    set_rng_seed(hyperpm['seed'])
    print(f'==========================run model {i + 1}==========================')
    log_param(hyperpm)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])
    dataset = Planetoid(root='../dataset', name=hyperpm['dataset'], transform=transform)
    train_data, val_data, test_data = dataset[0]

    model = GAE(encoder=Disen_Linear(train_data.x.size(1), hyperpm)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperpm['lr'])
    print(model)

    test_auc, test_ap = run_model()
    total_test_auc = total_test_auc + test_auc
    total_test_ap = total_test_ap + test_ap

    average_test_auc = total_test_auc/(i+1)
    average_test_ap = total_test_ap / (i+1)
    print(f'Model runs {i+1} times, average_test_auc:{average_test_auc}, average_test_ap:{average_test_ap}.')

# model.load_state_dict(torch.load('best_model_VGAE.pth'))
# z=model.encode(test_data.x, test_data.edge_index)
# visualize_dimreduc(z, test_data.y)
