import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
wo_disen=np.array([[-0.0752,  0.2417,  0.1753, -0.0883, -0.1708, -0.2458,  0.0852,  0.1289, -0.1113, -0.0778]])
wo_club=np.array([[-0.0413,  0.1445,  0.0704, -0.0187,  0.1651,  0.1104,  0.1986,  0.0901, 0.0608, -0.0294]])
vgdae=np.array([[-0.0268,  0.1401,  0.0317,  0.0122,  0.0421, -0.0239,  0.0013,  0.0389,0.0428, -0.1436]])


k=5
mat=np.vstack((wo_disen, wo_club, vgdae))
mat = abs(mat)
print(mat)
def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    plt.rc('font', family='Times New Roman')
    plt.rcParams['font.size'] = 9
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.tick_params(labelsize=9)

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="3%", pad="5%")
    cbar = ax.figure.colorbar(im, cax=cax , **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", family='Times New Roman', fontsize=9)


    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, family='Times New Roman', fontsize=9)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, family='Times New Roman', fontsize=9)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False, labelsize=9)

    return im, cbar
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",textcolors=("black", "white"),threshold=None, **textkw):
    plt.rc('font', family='Times New Roman')
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

fig, ax = plt.subplots()
yticks = ['w/o_disent', 'w/o_CLUB', 'VGDAE']
xticks = ['1-2', '1-3', '1-4', '1-5', '2-3', '2-4', '2-5', '3-4', '3-5', '4-5']
im, cbar = heatmap(mat, yticks , xticks, ax=ax, cmap="YlGn", cbar_kw=dict(),cbarlabel="correlation")
annotate_heatmap(im, valfmt="{x:.4f}", size=8)

fig.tight_layout()

plt.savefig('abl_fig.eps', bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.savefig('abl_fig.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
