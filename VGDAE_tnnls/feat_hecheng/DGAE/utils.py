from loguru import logger
import os
import random
import logging
#读数据
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.transforms as T

def dataloader(args):
    sizes =args.k*[args.num_c]
    probs  = [args.k*[args.q] for _ in range(args.k)]
    probs[0][0] = args.p1
    probs[1][1] = args.p2
    probs[2][2] = args.p3
    probs[3][3] = args.p4
    probs[4][4] = args.p5

    print(probs)
    G = nx.stochastic_block_model(sizes,probs,seed=0)

    x = torch.FloatTensor(nx.adjacency_matrix(G).todense())
    adj = nx.to_scipy_sparse_array(G).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)

    data = Data(x=x, edge_index=edge_index)


    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(args.device),
        T.RandomLinkSplit(num_val=args.val_frac, num_test=args.test_frac, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])

    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data, data


def get_logger(name):
    """ create a nice logger """
    logger = logging.getLogger(name)
    # clear handlers if they were created in other runs
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    # 定义handler的输出格式（formatter）
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # 创建一个handler，用于输出到控制台,决定将日志记录分配至正确的目的地
    ch = logging.StreamHandler() # 将日志写入控制台
    ch.setLevel(logging.DEBUG) # 指定将被分派到相应目标的最低严重性
    #给handler添加formatter
    ch.setFormatter(formatter)
    #给logger添加handler
    logger.addHandler(ch)

    # 创建一个handler，用于写入日志文件
    if name is not None:
        fh = logging.FileHandler(f'{name}.log')
        #给handler添加formatter
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        # 给logger添加handler
        logger.addHandler(fh)
    return logger


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


