import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 6
matplotlib.rcParams['font.family'] = 'Times New Roman'
barWidth = 0.29
# 设置柱子的高度
GAE=  [67.1, 77.0, 75.8, 86.9, 87.8, 94.1, 96.2]
VGAE= [73.1, 71.4, 80.9, 87.5, 85.3, 95.2, 95.8]
ARGVA=[65.1, 77.3, 78.20, 84.0, 81.1, 93.4, 88.7]
VGNAE=[70.7, 77.2, 72.0, 81.1, 79.5, 94.1, 94.1]
DGAE= [85.2, 80.5, 75.0, 94.1, 95.3, 96.4, 96.8]
VDGAE=[78.8, 82.8, 79.6, 93.0, 94.7, 96.5, 96]

r1 = np.arange(0,14,2)
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]

# 创建柱子
fig=plt.figure(figsize=(8,1.6))
plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
ax = plt.gca()

ax.spines['bottom'].set_linewidth(0.2);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(0.2);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(0.2);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(0.2);####设置上部坐标轴的粗细
ax.spines['bottom'].set_color('lightgray')
ax.spines['left'].set_color('lightgray')
ax.spines['right'].set_color('lightgray')
ax.spines['top'].set_color('lightgray')
plt.tick_params(axis='both', color='lightgray')

rects1=plt.bar(r1, GAE, width=barWidth, edgecolor='white', color='#76da91',label='GAE')
plt.bar_label(rects1, fontsize=5)
rects2=bar2=plt.bar(r2, VGAE,  width=barWidth, edgecolor='white', color='#f8cb7f', label='VGAE')
plt.bar_label(rects2, fontsize=5)
rects3=plt.bar(r3, ARGVA,  width=barWidth, edgecolor='white', color='#63b2ee', label='ARGVA')
plt.bar_label(rects3, fontsize=5)
rects4=plt.bar(r4, VGNAE,  width=barWidth, edgecolor='white', color='#f89588', label='VGNAE')
plt.bar_label(rects4, fontsize=5)
rects5=plt.bar(r5, DGAE, width=barWidth, edgecolor='white', color='#7cd6cf', label='DGAE')
plt.bar_label(rects5, fontsize=5)
rects6=plt.bar(r6, VDGAE, width=barWidth, edgecolor='white', color='#9192ab', label='VDGAE')
plt.bar_label(rects6, fontsize=5)

# 添加x轴名称


plt.xticks([r + 2*barWidth for r in r1], ['Wisconsin', 'Cornell', 'Texas', 'Computers', 'Photo', 'Squirrel', 'Chameleon'])
plt.ylabel('AUC(%)')
plt.ylim(56.5, 103.5)
plt.yticks(np.arange(60, 101, 20))
plt.grid(axis='y',linestyle='-',linewidth=0.2)

plt.legend(loc='lower center', ncols=6,bbox_to_anchor=(0.45,-0.31,0.1,0.1), borderaxespad = 1, handlelength=3, handletextpad=1.2, columnspacing=6)


plt.savefig('results.eps', dpi=600, bbox_inches='tight', pad_inches=0.1, format='eps')
plt.savefig('results.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.show()