import matplotlib.pyplot as plt
import numpy as np
#[0.0001, 0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2,0.3, 0.4, 0.45, 0.5, 0.55,0.57,0.59, 0.6,0.7,0.75,0.8,1]

Cora_lambda=[0.9503652841438954, 0.9473551555653172, 0.9515102852060822, 0.9523456318929604, 0.9520971882662596,  0.9577485194679669, 0.9580612950131373, 0.9584607684785332, 0.9559780079255826, 0.9594297954125578, 0.9598857947758535, 0.9600120035311777,  0.9620835255318574, 0.9636910524288063, 0.963463065074227, 0.9633475150236381,0.9631157891330049, 0.9611793877745187, 0.9621848503268724, 0.9627515123517102]
Citeseer_lambda=[0.9665885762589059, 0.9636710542205047, 0.9701823451274001, 0.9705397898804492, 0.9703707281729259,  0.9727630075188204, 0.9726125112688341, 0.9805627339693274, 0.9804661272793141, 0.9805289216278227, 0.9803743509238014, 0.980079700519261, 0.9797319164352131, 0.9809167854123898, 0.9809105180533753, 0.9735231722817059, 0.9733230234470018, 0.9732335823828244, 0.9732332820042047, 0.9731427276096427]
Pubmed_lambda=[0.9487889194437567, 0.9535557214759087, 0.9552192975846813, 0.957552742729281, 0.9557409946858425, 0.9561508431867352, 0.9556444973543249, 0.9571944155729906, 0.9581515942065582, 0.9601695527033455, 0.9616870688967013, 0.9618182122714357, 0.9638136749957643, 0.9658945322172842, 0.9689979617313402, 0.970013832177857, 0.9685046886632707, 0.967276811507338, 0.9659699091263122, 0.9620428291507041]

Cora_lambda=[i*100 for i in Cora_lambda]
Citeseer_lambda=[i*100 for i in Citeseer_lambda]
Pubmed_lambda=[i*100 for i in Pubmed_lambda]

def visualize_lambda(list1, list2, list3):
    plt.figure(dpi=600)
    #修改图中字体
    plt.rc('font', family='Times New Roman')
    #修改字号
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.tick_params(labelsize=9)
    #设置坐标轴名称
    plt.xlabel('$\lambda _{mi}$')
    plt.ylabel('AUC(%)')
    x=np.linspace(0,1,20)
    plt.xlim(0, 1)
    plt.ylim(94.5, 98.5)
    plt.xticks(np.arange(0, 1.1, step=0.1))
    plt.plot(x, list1, 'D', linestyle='-', color='#ff7f0e', markersize=6, linewidth=1.5, label='Cora')
    plt.plot(x, list2, 'o', linestyle='-', color='#1f77b4', markersize=6,linewidth=1.5, label='Citeseer')
    plt.plot(x, list3, '^', linestyle='-', color='#2ca02c', markersize=6, linewidth=1.5, label='Pubmed')

    plt.grid(visible=True, linestyle = ':')
    plt.legend(loc='lower right')
    plt.savefig("lambda.eps", dpi=600, format='eps')
    plt.savefig("lambda.png", dpi=600)
    # plt.show()

visualize_lambda(Cora_lambda,Citeseer_lambda,Pubmed_lambda)
