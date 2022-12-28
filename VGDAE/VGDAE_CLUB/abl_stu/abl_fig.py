import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# wo_disen=np.array([[-0.0752,  0.2417,  0.1753, -0.0883, -0.1708, -0.2458,  0.0852,  0.1289, -0.1113, -0.0778]])
# wo_club=np.array([[-0.0013,  0.1445,  0.0704, -0.0187,  0.1651,  0.0104,  0.1986,  0.0601, 0.0608, -0.0294]])
# vgdae=np.array([[-0.0368,  0.1401,  0.0317,  0.0122,  0.0421, -0.0239,  0.0013,  0.0389,0.0428, -0.1436]])
wo_disen=np.array([-0.1554,  0.4991,  0.3621, -0.1822, -0.3526, -0.5077,  0.1759,  0.2661,-0.2297, -0.1606])
wo_club=np.array([[-0.0040,  0.4543,  0.2214, -0.0588,  0.5190,  0.0328,  0.6244,  0.1891,0.1910, -0.0925]])
vgdae=np.array([-0.1670,  0.6364,  0.1441,  0.0554,  0.1913, -0.1088,  0.0058,  0.1769, 0.1944, -0.6525])

k=5
mat=np.vstack((wo_disen, wo_club, vgdae))
mat = abs(mat)
print(mat)
def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad="3%")
    cbar = ax.figure.colorbar(im, cax=cax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")


    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar
fig, ax = plt.subplots()
yticks = ['wo_disen', 'wo_club', 'vgdae']
xticks = np.arange(0, k*(k-1)/2)
im, cbar = heatmap(mat, yticks , xticks, ax=ax, cmap="YlGn", cbarlabel="correlation")

fig.tight_layout()

plt.savefig('abl_fig.eps', dpi=600)
plt.savefig('abl_fig.png', dpi=600)
