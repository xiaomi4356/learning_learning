import matplotlib.pyplot as plt
import matplotlib
import numpy as np
#number of channels
cora_auc=[92.8, 93.8, 95.4]
citeseer_auc=[94.1, 96.3, 97.4]
pubmed_auc=[93.5, 96.0, 97.8]
cora_ap=[92.8, 94.3, 95.8]
citeseer_ap=[94.1, 96.5, 98.0]
pubmed_ap=[92.8, 96.9, 97.8]

matplotlib.rcParams['font.size'] = 9.0

matplotlib.rcParams['font.family'] = 'Times New Roman'

fig, axs = plt.subplots(2, 3)
method = ['w/o dise', 'w/o CLUB', 'GDAE']
bar_c1 = ['violet', 'lightseagreen', 'lightsalmon']
bar_c2 = ['pink', 'lightgreen', 'lightcoral']

ax1=axs[0, 0].bar(method, cora_auc, width=0.65, color=bar_c1)
axs[0, 0].set_xlabel('Cora')
axs[0, 0].set_ylabel('AUC(%)')
axs[0, 0].set_ylim(90, 100)
axs[0, 0].bar_label(ax1, padding=3)
axs[0, 0].spines['right'].set_visible(False)
axs[0, 0].spines['top'].set_visible(False)

ax2=axs[0, 1].bar(method, citeseer_auc, width=0.65, color=bar_c1)
axs[0, 1].set_xlabel('Citeseer')
axs[0, 1].set_ylabel('AUC(%)')
axs[0, 1].set_ylim(90, 100)
axs[0, 1].bar_label(ax2, padding=3)
axs[0, 1].spines['right'].set_visible(False)
axs[0, 1].spines['top'].set_visible(False)

ax3=axs[0, 2].bar(method, pubmed_auc, width=0.65, color=bar_c1)
axs[0, 2].set_xlabel('Pubmed')
axs[0, 2].set_ylabel('AUC(%)')
axs[0, 2].set_ylim(90, 100)
axs[0, 2].bar_label(ax3, padding=3)
axs[0, 2].spines['right'].set_visible(False)
axs[0, 2].spines['top'].set_visible(False)

ax4=axs[1, 0].bar(method, cora_ap, width=0.65, color=bar_c2)
axs[1, 0].set_xlabel('Cora')
axs[1, 0].set_ylabel('AP(%)')
axs[1, 0].set_ylim(90, 100)
axs[1, 0].bar_label(ax4, padding=3)
axs[1, 0].spines['right'].set_visible(False)
axs[1, 0].spines['top'].set_visible(False)

ax5=axs[1, 1].bar(method, citeseer_ap, width=0.65, color=bar_c2)
axs[1, 1].set_xlabel('Citeseer')
axs[1, 1].set_ylabel('AP(%)')
axs[1, 1].set_ylim(90, 100)
axs[1, 1].bar_label(ax5, padding=3)
axs[1, 1].spines['right'].set_visible(False)
axs[1, 1].spines['top'].set_visible(False)

ax6=axs[1, 2].bar(method, pubmed_ap, width=0.65, color=bar_c2)
axs[1, 2].set_xlabel('Pubmed')
axs[1, 2].set_ylabel('AP(%)')
axs[1, 2].set_ylim(90, 100)
axs[1, 2].bar_label(ax6, padding=3)
axs[1, 2].spines['right'].set_visible(False)
axs[1, 2].spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('abl_gdae.eps', dpi=600, format='eps')
plt.savefig('abl_gdae.png', dpi=600)
plt.show()