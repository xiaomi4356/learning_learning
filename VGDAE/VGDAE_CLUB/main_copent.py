import random
import torch
import numpy as np
from torch_geometric.nn import GAE, VGAE, GCNConv
from model import Disen_Linear
from copent import copent
from utils import *
from correlation import *
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hyperpm=dict()
hyperpm['dataset']='Cora'
hyperpm['epoch']=200
hyperpm['model_lr']=0.005
hyperpm['k']=4
hyperpm['out_dim']=256
hyperpm['hidden_dim']=64
hyperpm['x_dim']=64
hyperpm['y_dim']=64
hyperpm['run_num']=1
hyperpm['routit']=3
hyperpm['dropout']=0
hyperpm['model']='GAE'
hyperpm['alpha']=1
hyperpm['norm']=1

#面临的问题是两个模型，两个优化器
def train():
    model.train()
    total_cor = 0
    z = model.encode(train_data.x, train_data.edge_index, hyperpm)

    for i in range(hyperpm['k']):
        for j in range(i + 1, hyperpm['k']):
            h_x = copent(z[:, i * hyperpm['hidden_dim']: (i + 1) * hyperpm['hidden_dim']].cpu().detach().numpy())
            h_y = copent(z[:, j * hyperpm['hidden_dim']: (j + 1) * hyperpm['hidden_dim']].cpu().detach().numpy())
            h_xy = copent(torch.cat([z[:, i * hyperpm['hidden_dim']: (i + 1) * hyperpm['hidden_dim']],
                              z[:, j * hyperpm['hidden_dim']: (j + 1) * hyperpm['hidden_dim']]]).cpu().detach().numpy())
            cor = h_xy-h_x-h_y


            total_cor = total_cor+cor

    recon_loss = model.recon_loss(z, train_data.pos_edge_label_index)
    loss = recon_loss + hyperpm['alpha']*total_cor
    if hyperpm['model']=='VGAE':
        # loss = loss+(1 / train_data.num_nodes)*model.kl_loss()
        loss = loss+(1 / train_data.edge_index.size(1))*model.kl_loss()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, total_cor

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index, hyperpm)
    auc, ap=model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    return auc, ap

def run_model():
    step, best_auc, best_ap, best_model = 0, 0, 0, None,

    for epoch in range(hyperpm['epoch']):
        loss, total_cor = train()

        # for name, param in model.named_parameters():
        #     writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        #     writer.add_histogram(name + '_grads', param.grad.clone().cpu().data.numpy(), epoch)

        val_auc, val_ap = test(val_data)

        writer.add_scalar('train_loss', loss.cpu().data.numpy(), epoch)
        writer.add_scalar('total_cor', total_cor, epoch)
        writer.add_scalar('val_auc', val_auc, epoch)
        writer.add_scalar('val_ap', val_ap, epoch)

        print(f'Epoch: {epoch:03d}, train_loss:{loss:.6f}, val_auc: {val_auc:.6f}, val_ap: {val_ap:.6f}, total_cor:{total_cor:.6f}')
        if (val_auc+val_ap) > (best_auc+best_ap):
            best_ap = val_ap
            best_auc =val_auc
            torch.save(model.state_dict(), 'club_vgae_pubmed.pth')
            step = epoch

    model.load_state_dict(torch.load('club_vgae_pubmed.pth'))

    test_auc, test_ap = test(test_data)
    print(f'the epoch:{step}, test_auc:{test_auc:.6f}, test_ap:{test_ap:.6f}')

    return test_auc, test_ap

total_test_auc, total_test_ap = 0, 0
for i in range(hyperpm['run_num']):
    hyperpm['seed'] = 2102048
    set_rng_seed(hyperpm['seed'])
    print(f'==========================run model {i + 1}==========================')
    log_param(hyperpm)
    writer = SummaryWriter('./logs_club')

    train_data, val_data, test_data = dataloader(hyperpm, device)
    model = eval(hyperpm['model'])(encoder=Disen_Linear(train_data.x.size(1), hyperpm)).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), hyperpm['model_lr'])

    test_auc, test_ap = run_model()

    total_test_auc = total_test_auc + test_auc
    total_test_ap = total_test_ap + test_ap

    average_test_auc = total_test_auc/(i+1)
    average_test_ap = total_test_ap / (i+1)
    log_param(hyperpm)
    print(f'Model runs {i+1} times, average_test_auc:{average_test_auc:.6f}, average_test_ap:{average_test_ap:.6f}.')

