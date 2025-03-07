import torch
from torch import mean
def DistanceCorrelation(tensor_1, tensor_2):
    # ref: https://en.wikipedia.org/wiki/Distance_correlation
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








