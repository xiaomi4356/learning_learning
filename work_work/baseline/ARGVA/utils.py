import torch
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
import os
import random
from sklearn.manifold import TSNE
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, webkb

#读数据
def dataloader(hyperpm, device):
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])
    dataset = Planetoid(root='../dataset', name=hyperpm['dataset'], transform=transform)
    train_data, val_data, test_data = dataset[0]
    return train_data, val_data, test_data


def log_param(hyperpm):
    for key, value in hyperpm.items():
        if type(value) is dict():
            for in_key, in_value in value:
                logger.info(f'{key:20}:{value:>20}')
        else:
            if (value != None):
                logger.info(f'{key:20}:{value:>20}')

def set_rng_seed(seed):
    random.seed(seed) #为python设置随机种子
    np.random.seed(seed)  #为numpy设置随机种子
    torch.manual_seed(seed)   #为CPU设置随机种子
    torch.cuda.manual_seed(seed)   #为当前GPU设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)   #为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # torch.use_deterministic_algorithms(True)
    warn_only = True
