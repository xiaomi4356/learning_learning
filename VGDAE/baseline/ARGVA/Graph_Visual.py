import torch
from torch_geometric.nn import GAE, VGAE, GCNConv, ARGVA
import torch.nn.functional as F
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hyperpm=dict()
hyperpm['dataset']='Citeseer'
hyperpm['k']=4
hyperpm['hidden_dim']=64
hyperpm['x_dim']=64
hyperpm['mi_iter']=5
hyperpm['routit']=3
hyperpm['dropout']=0
hyperpm['model']='VGAE'
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1,hidden_channels2, out_channels):
        super(Discriminator, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels1)
        self.lin2 = torch.nn.Linear(hidden_channels1, hidden_channels2)
        self.lin3 = torch.nn.Linear(hidden_channels2, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return x

train_data, val_data, test_data = dataloader(hyperpm, device)
encoder = Encoder(train_data.x.size(1), hidden_channels=64, out_channels=32)
discriminator = Discriminator(in_channels=32, hidden_channels1=16, hidden_channels2=64,
                              out_channels=128)
model = ARGVA(encoder, discriminator).to(device)
model.load_state_dict(torch.load('arvga_cora.pth'))
z = model.encode(train_data.x, train_data.edge_index)

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