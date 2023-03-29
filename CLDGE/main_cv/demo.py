import torch
import numpy as np
from loguru import logger
import os
import random
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, Amazon, WikipediaNetwork
import logging
#读数据
from sklearn.model_selection import train_test_split, GridSearchCV

dataset = Planetoid(root='../dataset', name='Cora')
data = dataset[0]
print(data)
X_train, X_test, y_train, y_test = train_test_split(data.x, data.y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)