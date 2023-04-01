import matplotlib.pyplot as plt
import matplotlib
import numpy as np
#number of channels
cora_auc=[92.8, 93.8, 95.4]
citeseer_auc=[94.1, 96.3, 97.4]
pubmed_auc=[93.5, 96.0, 97.8]


matplotlib.rcParams['font.size'] = 8

matplotlib.rcParams['font.family'] = 'Times New Roman'

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(5,2.8))
plt.subplots_adjust(wspace=0, hspace=0.1)  # 调整子图间距
method = ['w/o \n disen', 'w/o \n CLUB', 'GDAE']
# bar_c1 = ['violet',  'lightgreen', 'lightsalmon', 'pink', 'lightseagreen','lightcoral','cornflowerblue','dodgerblue',]
bar_c1 = ['cornflowerblue','lightcoral','lightseagreen']

ax_0=ax0.bar(method, cora_auc, width=0.55, color=bar_c1)
ax0.set_xlabel('Cora', fontsize=10)
ax0.set_ylabel('AUC(%)', labelpad=-20, y=1.05, rotation=0, fontsize=10)
ax0.set_ylim(90, 100)
ax0.bar_label(ax_0, padding=0)
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)

ax_1=ax1.bar(method, citeseer_auc, width=0.6, color=bar_c1)
ax1.set_xlabel('Citeseer',fontsize=10)
ax1.set_ylabel('AUC(%)', labelpad=-20, y=1.05, rotation=0, fontsize=10)
ax1.set_ylim(90, 100)
ax1.bar_label(ax_1, padding=0)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax_2=ax2.bar(method, pubmed_auc, width=0.6, color=bar_c1)
ax2.set_xlabel('Pubmed',fontsize=10)
ax2.set_ylabel('AUC(%)', labelpad=-20, y=1.05, rotation=0, fontsize=10)
ax2.set_ylim(90, 100)
ax2.bar_label(ax_2, padding=0)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)



plt.tight_layout()
plt.savefig('abl_gdae.eps', dpi=600, bbox_inches='tight', pad_inches=0.1, format='eps')
plt.savefig('abl_gdae.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.show()