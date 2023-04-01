import torch
from torch_geometric.nn import GAE, VGAE, GCNConv
from model import Disen_Linear
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hyperpm=dict()
hyperpm['dataset']='Cora'
hyperpm['k']=6
hyperpm['hidden_dim']=64
hyperpm['x_dim']=64
hyperpm['mi_iter']=5
hyperpm['routit']=3
hyperpm['dropout']=0
hyperpm['model']='VGAE'

train_data, val_data, test_data = dataloader(hyperpm, device)
# #VGDAE
model = eval(hyperpm['model'])(encoder=Disen_Linear(train_data.x.size(1), hyperpm)).to(device)
model.load_state_dict(torch.load('vgdae_cora.pth'))
z = model.encode(train_data.x, train_data.edge_index, hyperpm)
print(z.shape)
def visualize_dimreduc(h, color):
    color = color.cpu()
    z = TSNE(n_components=2, perplexity=50, early_exaggeration=20,
             n_iter=5000, method='barnes_hut',angle=0.4, learning_rate=10,
             init='pca', random_state=0).fit_transform(h.cpu().detach().numpy())  # detach:返回与张量相同的tensor，但没有梯度  #TSNE：将高维数据降维，默认是降成两维
    plt.figure(figsize=(10, 10))  # 规范画布尺寸
    plt.xticks([])  # 空列表意味着不显示坐标轴刻度
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=40, c=color, cmap='Set2')
    plt.show()
    # 以降维后的数据画散点图，点的大小s=70，点的颜色有color种，颜色条是set2
visualize_dimreduc(z, train_data.y)