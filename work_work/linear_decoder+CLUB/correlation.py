import torch
from torch import mean
from torch import nn
def DistanceCorrelation(tensor_1, tensor_2):
    # ref: https://en.wikipedia.org/wiki/Distance_correlation
    #输入：tensor_1, tensor_2分别是两个矩阵，分别计算tensor_1和tensor_2每一行之间的相关性，然后将所有行的相关性先加和再求平均
    n, d = tensor_1.size(0), tensor_1.size(1)
    """cul distance matrix"""
    a_xj, a_xk = tensor_1.unsqueeze(-1).repeat(1,1,d), tensor_1.unsqueeze(1).repeat(1,d,1)
    b_xj, b_xk = tensor_2.unsqueeze(-1).repeat(1,1,d), tensor_2.unsqueeze(1).repeat(1,d,1)
    a, b = abs(a_xj-a_xk), abs(b_xj-b_xk)
    a_col_mean, a_row_mean = torch.mean(a, dim=1, keepdim=True),torch.mean(a, dim=-1, keepdim=True)
    a_mean = torch.mean(a_col_mean, dim=-1, keepdim=True)
    b_col_mean, b_row_mean = torch.mean(b, dim=1, keepdim=True), torch.mean(b, dim=-1, keepdim=True)
    b_mean = torch.mean(b_col_mean, dim=-1, keepdim=True)
    """cul distance correlation"""
    A = a - a_col_mean - a_row_mean + a_mean
    B = b - b_col_mean - b_row_mean + b_mean
    dcov_AB = torch.sqrt(((A * B).sum(dim=1).sum(dim=1) / d ** 2) + 1e-8)
    dcov_AA = torch.sqrt(((A * A).sum(dim=1).sum(dim=1) / d ** 2) + 1e-8)
    dcov_BB = torch.sqrt(((B * B).sum(dim=1).sum(dim=1) / d ** 2) + 1e-8)
    cor = dcov_AB / torch.sqrt(dcov_AA * dcov_BB )
    mean_cor = cor.sum()/ n
    return mean_cor

# 调试代码
# tensor1=torch.randint(0,10,(5,4), dtype=float)
# tensor2=torch.randint(0,10,(5,4), dtype=float)
# mean_cor=DistanceCorrelation(tensor1, tensor2)
# print(f'tensor1:{tensor1}')
# print(f'tensor2:{tensor2}')

class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.Sigmoid(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.Sigmoid(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Sigmoid())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
