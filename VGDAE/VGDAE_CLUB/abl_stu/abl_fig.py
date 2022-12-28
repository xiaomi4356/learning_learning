import numpy as np
import matplotlib.pyplot as plt
wo_disen=np.array([[-0.5828, -0.2487,  0.3050,  0.0853, -0.6698,  0.2226]])
wo_club=np.array([[-0.2376, -0.0248, -0.2334,  0.5188, -0.3676, -0.0269]])
vgdae=np.array([[-0.2592, -0.0106, -0.1334,  0.5059, -0.3012, -0.0315]])
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
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
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
xticks = np.arange(0, 6)
im, cbar = heatmap(mat, yticks , xticks, ax=ax, cmap="YlGn", cbarlabel="correlation")

fig.tight_layout()

plt.savefig('abl_fig.eps', dpi=600)
plt.savefig('abl_fig.png', dpi=600)
