import numpy as np
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.transforms as T


sizes =[500,500,500,500,500]
probs  = \
[[0.01, 1e-06, 1e-06, 1e-06, 1e-06], [1e-06, 0.02, 1e-06, 1e-06, 1e-06], [1e-06, 1e-06, 0.05, 1e-06, 1e-06], [1e-06, 1e-06, 1e-06, 0.04, 1e-06], [1e-06, 1e-06, 1e-06, 1e-06, 0.08]]

print(probs)
G = nx.stochastic_block_model(sizes, probs, seed=0)
d = dict(nx.degree(G))
print(d)
print("平均度为：", sum(d.values())/len(G.nodes))


