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
def dataloader(dataname, device):
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])
    dataset = Planetoid(root='../dataset', name=dataname, transform=transform)
    train_data, val_data, test_data = dataset[0]
    return train_data, val_data, test_data


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
    torch.use_deterministic_algorithms(True)

    # 以降维后的数据画散点图，点的大小s=70，点的颜色有color种，颜色条是set2

