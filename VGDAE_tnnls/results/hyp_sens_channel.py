import matplotlib.pyplot as plt
import matplotlib
import numpy as np
#number of channels
Cora_channel=[0.9376521717213542, 0.9479139736937807, 0.9506101271383255, 0.9513295334660766, 0.9550849929247576, 0.9545312156814738, 0.9522397733041922, 0.9575823914679418, 0.9564950005220917, 0.9589657543864704]
Citeseer_channel=[0.9584272430865838, 0.9632884917280521, 0.9740388841927302, 0.9678415650283781, 0.9733404178239343, 0.9760550658133076, 0.9762647023306364, 0.9750677454413719, 0.9756618765849534, 0.9781528800869459]
Pubmed_channel=[0.9436856705091946, 0.9590673170753561, 0.963982133955219, 0.9665333307810606, 0.9700496074251586, 0.9700972028991645, 0.9697850399213465, 0.9666842983910907, 0.970017422315878, 0.9689330508999205]

Cora_channel=[i*100 for i in Cora_channel]
Citeseer_channel=[i*100 for i in Citeseer_channel]
Pubmed_channel=[i*100 for i in Pubmed_channel]
def visualize_Channels(list1, list2, list3):
    plt.figure(dpi=600)
    matplotlib.rcParams['font.size'] = 18

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    #修改图中字体
    # plt.rc('font', family='Times New Roman')
    #修改字号
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.tick_params(labelsize=12)
    #设置坐标轴名称
    plt.xlabel('Number of channels')
    plt.ylabel('AUC(%)')
    plt.xlim(1, 11)
    plt.ylim(93, 98.4)
    plt.xticks(np.arange(0, 11, step=1))
    x = range(1, 11)
    plt.plot(x, list1, 'D', linestyle='-', color='#ff7f0e', markersize=6, linewidth=1.5, label='Cora')
    plt.plot(x, list2, 'o', linestyle='-', color='#1f77b4', markersize=6, linewidth=1.5, label='Citeseer')
    plt.plot(x, list3, '^', linestyle='-', color='#2ca02c', markersize=6, linewidth=1.5, label='Pubmed')
    plt.grid(visible=True, linestyle = ':')
    plt.legend(loc='lower right')
    plt.savefig("channels.eps", dpi=600,bbox_inches='tight', pad_inches=0.1, format='eps')
    plt.savefig("channels.png", dpi=600,bbox_inches='tight', pad_inches=0.1,)
    # plt.show()

visualize_Channels(Cora_channel, Citeseer_channel, Pubmed_channel)