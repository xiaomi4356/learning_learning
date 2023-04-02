import matplotlib.pyplot as plt
import numpy as np
import matplotlib
#[0.0001, 0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2,0.3, 0.4, 0.45, 0.5, 0.55,0.57,0.59, 0.6,0.7,0.75,0.8,1]

Cora_lambda=[0.95, 0.9562199122165852, 0.9581203259292332, 0.9494312801327913, 0.9564877992575497, 0.9494986119562595, 0.9522602969081371, 0.9560333994649459, 0.9518242603401157, 0.9555379524644527, 0.9555595562580788]
Citeseer_lambda=[0.965, 0.9738727206859075, 0.9753759207825141, 0.9747711629030311, 0.9720458881777564, 0.9690385219176427, 0.9686144185484846, 0.9763748339572516, 0.9702586644125105, 0.9744040574809807, 0.9713174737350562]
Pubmed_lambda=[0.957,0.9677385210855739, 0.9639840227049096, 0.9602818493089315, 0.9616156120485735, 0.9612570601565249, 0.9588336059377811, 0.9568182692902945, 0.9573794824968396, 0.9571621082804416, 0.9556282958936647]

Cora_lambda=[i*100 for i in Cora_lambda]
Citeseer_lambda=[i*100 for i in Citeseer_lambda]
Pubmed_lambda=[i*100 for i in Pubmed_lambda]

def visualize_lambda(list1, list2, list3):
    plt.figure(dpi=600)
    matplotlib.rcParams['font.size'] = 18

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    #修改图中字体
    # plt.rc('font', family='Times New Roman')
    #修改字号
    # plt.xticks(fontsize=9)
    # plt.yticks(fontsize=9)
    # plt.tick_params(labelsize=9)
    #设置坐标轴名称
    plt.xlabel('$\lambda _{mi}$')
    plt.ylabel('AUC(%)')
    x=np.linspace(0,1,11)
    plt.xlim(-0.05, 1.05)
    plt.ylim(93, 98)
    plt.xticks(np.arange(0, 1.1, step=0.1))
    plt.yticks(np.arange(93, 98.5, step=1))
    plt.plot(x, list1, 'D', linestyle='-', color='#ff7f0e', markersize=6, linewidth=1.5, label='Cora')
    plt.plot(x, list2, 'o', linestyle='-', color='#1f77b4', markersize=6,linewidth=1.5, label='Citeseer')
    plt.plot(x, list3, '^', linestyle='-', color='#2ca02c', markersize=6, linewidth=1.5, label='Pubmed')

    plt.grid(visible=True, linestyle = ':')
    plt.legend(loc='lower right')
    plt.savefig("lambda.eps", dpi=600,bbox_inches='tight', pad_inches=0.1, format='eps')
    plt.savefig("lambda.png", dpi=600,bbox_inches='tight', pad_inches=0.1,)
    # plt.show()

visualize_lambda(Cora_lambda,Citeseer_lambda,Pubmed_lambda)
