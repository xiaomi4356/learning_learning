import numpy as np
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.transforms as T


sizes =[500,500,500,500]
probs  = [[0.01, 3e-05, 3e-05, 3e-05], [3e-05, 0.02, 3e-05, 3e-05], [3e-05, 3e-05, 0.03, 3e-05], [3e-05, 3e-05, 3e-05, 0.04]]

print(probs)
G = nx.stochastic_block_model(sizes, probs, seed=0)

x = torch.FloatTensor(nx.adjacency_matrix(G).todense())
adj = nx.to_scipy_sparse_array(G).tocoo()
row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
edge_index = torch.stack([row, col], dim=0)

data = Data(x=x, edge_index=edge_index)


transform = T.Compose([
    T.NormalizeFeatures(),
    # T.ToDevice(args.device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      split_labels=True, add_negative_train_samples=False),
])

train_data, val_data, test_data = transform(data)
print(train_data)
