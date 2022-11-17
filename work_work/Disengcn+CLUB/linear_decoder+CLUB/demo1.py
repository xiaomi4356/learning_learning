
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
from torch.nn import Module, ModuleList
from torch.nn.functional import cross_entropy
from torch_geometric import nn
import torch
# 下载数据
dataset = Planetoid(root='./dataset', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

# 搭建模型
class GCN(Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = nn.GCNConv(dataset.num_features, dataset.num_classes)
        self.conv2 = nn.GCNConv(dataset.num_features, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x

    def learning(self, x, edge_index):
        x = self.conv2(x, edge_index)
        return x

class basemodel(Module):
    def __init__(self):
        super(basemodel, self).__init__()
        self.GCN_list=ModuleList([GCN() for i in range(2)])
        self.gcn=nn.GCNConv(dataset.num_features, dataset.num_classes)
    def lld_best(self, x, edge_index, y):
        lld_loss=0
        for i in range(2):
            out=self.GCN_list[i].learning(x, edge_index)
            lld_loss += cross_entropy(out, y)
        return lld_loss
    def mi_l(self,x, edge_index, y):
        mi_loss = 0
        for i in range(2):
            out = self.GCN_list[i](x, edge_index)
            mi_loss += cross_entropy(out, y)
        return mi_loss
    def forward_base(self, x, edge_index, y):
        x_=self.gcn(x, edge_index)
        mi_loss=self.mi_l(x, edge_index, y)
        return x_, mi_loss

model = basemodel()
#list:将map对象转换成list对象

params=list(model.GCN_list.named_parameters())

GCN_list_params = list(map(id,model.GCN_list.parameters()))
print(f'GCN_list_params:{GCN_list_params}')


rest_params = filter(lambda x:id(x) not in GCN_list_params, model.parameters())
params_rest_params=list(model.named_parameters())

GCN_list_optimizer=torch.optim.Adam(model.GCN_list.parameters(), lr=0.01, weight_decay=4e-5)
optimizer=torch.optim.Adam(rest_params, lr=0.01, weight_decay=4e-5)


for i in range(2):
    model.train()
    model.GCN_list.eval()
    out, mi_loss=model.forward_base(data.x, data.edge_index, data.y)
    loss=cross_entropy(out, data.y)
    loss=loss+mi_loss
    optimizer.zero_grad()
    loss.backward()
    params_model = list(model.named_parameters())
    print('\n')
    print('\n')
    print(f'=============================epoch:{i}loss.backward====================================')
    print(f'params_model: {params_model}')
    optimizer.step()
    params_rest_params = list(model.named_parameters())
    print('\n')
    print('\n')
    print(f'=============================epoch:{i}optimizer.step====================================')
    print(f'params_model: {params_model}')


    model.GCN_list.train()
    lld_loss=model.lld_best(data.x, data.edge_index, data.y)
    GCN_list_optimizer.zero_grad()
    lld_loss.backward()
    params_rest_params = list(model.named_parameters())
    print('\n')
    print('\n')
    print(f'=============================epoch:{i} lld_loss.backward===================================')
    print(f'params_model: {params_model}')

    GCN_list_optimizer.step()
    params_rest_params = list(model.named_parameters())
    print('\n')
    print('\n')
    print(f'=============================epoch:{i}optimizer.step====================================')
    print(f'params_model: {params_model}')



