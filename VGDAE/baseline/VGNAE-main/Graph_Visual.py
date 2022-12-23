import torch
from utils import *
import torch.nn as nn
from torch_geometric.nn import GAE, VGAE, APPNP
from utils import *
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hyperpm=dict()
hyperpm['dataset']='Cora'
hyperpm['model']='VGAE'

train_data, val_data, test_data = dataloader(hyperpm, device)

class VGNAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGNAEEncoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)

    def forward(self, x, edge_index, hyperpm):
        if hyperpm['model'] == 'GAE':
            x = self.linear1(x)
            x = F.normalize(x,p=2,dim=1)  * 1.8
            x = self.propagate(x, edge_index)
            return x

        if hyperpm['model'] == 'VGAE':
            x_ = self.linear1(x)
            x_ = self.propagate(x_, edge_index)

            x = self.linear2(x)
            x = F.normalize(x,p=2,dim=1) * 0.4
            x = self.propagate(x, edge_index)
            return x, x_

        return x
model = eval(hyperpm['model'])(VGNAEEncoder(train_data.x.size(1), 128)).to(device)
print(model)
model.load_state_dict(torch.load('vgnae_cora.pth'))
z = model.encode(train_data.x, train_data.edge_index, hyperpm)

def visualize_dimreduc(h, color):
    color = color.cpu()
    z = TSNE().fit_transform(h.cpu().detach().numpy())  # detach:返回与张量相同的tensor，但没有梯度  #TSNE：将高维数据降维，默认是降成两维
    plt.figure(figsize=(10, 10))  # 规范画布尺寸
    plt.xticks([])  # 空列表意味着不显示坐标轴刻度
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap='Set2')
    plt.show()
    # 以降维后的数据画散点图，点的大小s=70，点的颜色有color种，颜色条是set2
visualize_dimreduc(z, train_data.y)