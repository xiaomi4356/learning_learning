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

# def dataloader_1(hyperpm, device):
#     transform = T.Compose([
#         T.NormalizeFeatures(),
#         T.ToDevice(device),
#         T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
#                           split_labels=True, add_negative_train_samples=False),
#     ])
#     dataset = webkb(root='../dataset', name='Cornell', transform=transform)
#     train_data, val_data, test_data = dataset[0]
#     return train_data, val_data, test_data

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
    torch.use_deterministic_algorithms(True)

def visualize(list1):
    # plt.figure(figsize=(12,18))
    list1 = torch.tensor(list1, device='cpu')
    plt.plot(list1)
    plt.ylabel('train_loss')
    plt.xlabel('epoch')
    plt.grid(True)

    plt.show()

    # 可视化
def visualize_dimreduc(h, color):
    color = color.cpu()
    z = TSNE().fit_transform(h.cpu().detach().numpy())  # detach:返回与张量相同的tensor，但没有梯度  #TSNE：将高维数据降维，默认是降成两维
    plt.figure(figsize=(10, 10))  # 规范画布尺寸
    plt.xticks([])  # 空列表意味着不显示坐标轴刻度
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap='Set2')
    plt.show()
    # 以降维后的数据画散点图，点的大小s=70，点的颜色有color种，颜色条是set2

