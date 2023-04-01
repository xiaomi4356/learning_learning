import matplotlib.pyplot as plt
import numpy as np
#number of channels
Cora_channel=[94.86908613537757, 95.58315473800899, 95.86344524315157, 96.2847224829022, 96.28085090484872, 96.29632690325019, 96.37078807538253, 96.25202576378918, 96.27582174803852, 96.34931122369561, 96.30672170790815, 96.32915112363227, 96.50758965102528, 96.4482268592424, 96.27215812439981, 96.38796965843814, 96.36134810601489, 96.30346340308606]
Citeseer_channel=[97.16734177003018, 97.44864178613608, 97.66442715573092, 97.60904346042712, 97.85273452020711, 97.75465601887052, 97.9822355095755, 97.97212699267719, 97.99240772153337, 98.078376499358, 97.9254617706015, 98.04907560285677, 98.00283842487654, 97.99425818718697, 97.95397123291576, 98.08436901425821, 98.01095336285502, 98.0105811883641]
Pubmed_channel=[95.76105005115407, 96.5538743703489, 96.87946223478086, 97.01448747377133, 96.96083120055651, 96.85133971917398, 96.78453092621433, 96.81194579705846, 96.77237369100993, 96.73598598639369, 96.77641337532094, 96.79851327399027, 96.76661325900247, 96.80729265092079, 96.84967497214222, 96.84369817393033, 96.77817230528875, 96.79509214247548]
def visualize_Channels(list1, list2, list3):
    plt.figure(dpi=600)
    #修改图中字体
    plt.rc('font', family='Times New Roman')
    #修改字号
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.tick_params(labelsize=9)
    #设置坐标轴名称
    plt.xlabel('Number of channels')
    plt.ylabel('AUC(%)')
    plt.xlim(2, 20)
    plt.ylim(94.75, 98.4)
    plt.xticks(np.arange(2, 22, step=2))
    x = range(2, 20)
    plt.plot(x, list1, 'D', linestyle='-', color='#ff7f0e', markersize=6, linewidth=1.5, label='Cora')
    plt.plot(x, list2, 'o', linestyle='-', color='#1f77b4', markersize=6, linewidth=1.5, label='Citeseer')
    plt.plot(x, list3, '^', linestyle='-', color='#2ca02c', markersize=6, linewidth=1.5, label='Pubmed')
    plt.grid(visible=True, linestyle = ':')
    plt.legend(loc='lower right')
    plt.savefig("channels.eps", dpi=600, format='eps')
    plt.savefig("channels.png", dpi=600)
    # plt.show()

visualize_Channels(Cora_channel, Citeseer_channel, Pubmed_channel)