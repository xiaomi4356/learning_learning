import random
import torch
from utils import *
from torch_geometric.nn import GCNConv, InnerProductDecoder
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from hyperspherical_vae import *

hyperpm=dict()
device=torch.device('cpu')
hyperpm['dataset']='Cora'
hyperpm['epoch']=2000
hyperpm['run_num']=1
EPS = 1e-15
MAX_LOGSTD = 10


class GCNEncoder(torch.nn.Module):
    def __init__(self, i_dim, z_dim_mu, z_dim_var):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(i_dim, 64, cached=True)
        self.conv_mu = GCNConv(64, z_dim_mu, cached=True)
        self.conv_var = GCNConv(64, z_dim_var, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.1, training=self.training)
        return self.conv_mu(x, edge_index), self.conv_var(x, edge_index)


class ModelVAE(torch.nn.Module):
    def __init__(self, i_dim, h_dim, z_dim, activation=F.relu, distribution='normal'):
        """
        ModelVAE initializer
        :param i_dim: dimension of the input data
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        super(ModelVAE, self).__init__()

        self.z_dim, self.activation, self.distribution = z_dim, activation, distribution

        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            self.encoder = GCNEncoder(i_dim, z_dim, z_dim)
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.encoder = GCNEncoder(i_dim, z_dim, 1)
        else:
            raise NotImplemented

        self.decoder = InnerProductDecoder()

    def encode(self, x, edge_index):
        # 2 hidden layers encoder
        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            z_mean, z_var = self.encoder(x, edge_index)
            z_var = F.softplus(z_var)

        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            z_mean, z_var = self.encoder(x, edge_index)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(z_var)
        else:
            raise NotImplemented

        return z_mean, z_var

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(self.decoder(z, pos_edge_index) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()
        return pos_loss + neg_loss

    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        elif self.distribution == 'vmf':
            # q_z = VonMisesFisher(z_mean, z_var,validate_args=False)
            # p_z = HypersphericalUniform(self.z_dim - 1,validate_args=False)
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1)
        else:
            raise NotImplemented

        return q_z, p_z

    def forward(self, x, edge_index):
        z_mean, z_var = self.encode(x, edge_index)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        return (q_z, p_z), z


def compute_scores(z, test_pos, test_neg):
    test = torch.cat((test_pos, test_neg), dim=1)
    labels = torch.zeros(test.size(1), 1)
    labels[0:test_pos.size(1)] = 1
    row, col = test
    src = z[row]
    tgt = z[col]
    scores = torch.sigmoid(torch.sum(src * tgt, dim=1))
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap


def train():
    model.train()
    optimizer.zero_grad()
    (q_z, p_z), z = model(train_data.x, train_data.edge_index)
    loss_recon = model.recon_loss(z, train_data.edge_index)
    if model.distribution == 'normal':
        loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
    elif model.distribution == 'vmf':
        loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
    else:
        raise NotImplemented

    loss = loss_recon + (1 / train_data.edge_index.size(1)) * loss_KL
    loss.backward()
    optimizer.step()

    return loss

@torch.no_grad()
def test(data):

    model.eval()
    _, z = model(data.x, data.edge_index)
    z = z.cpu().clone().detach()
    auc, ap= compute_scores(z, data.pos_edge_label_index, data.neg_edge_label_index)
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
            torch.save(model.state_dict(), 's_vgae_cora_bias.pth')
            step = epoch

    model.load_state_dict(torch.load('s_vgae_cora_bias.pth'))
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
    H_DIM = 16
    Z_DIM = 64
    distribution = 'vmf'
    if distribution == 'normal':
        model = ModelVAE(i_dim=train_data.x.size(1), h_dim=H_DIM, z_dim=Z_DIM, distribution='normal').to(device)
    if distribution == 'vmf':
        model = ModelVAE(i_dim=train_data.x.size(1), h_dim=H_DIM, z_dim=Z_DIM + 1, distribution='vmf').to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    print(model)
    test_auc, test_ap = run_model()
    total_test_auc = total_test_auc + test_auc
    total_test_ap = total_test_ap + test_ap

    average_test_auc = total_test_auc/(i+1)
    average_test_ap = total_test_ap / (i+1)
    log_param(hyperpm)
    print(f'Model runs {i+1} times, average_test_auc:{average_test_auc}, average_test_ap:{average_test_ap}.')
