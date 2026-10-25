"""Draws figures/pipeline.png: the final two-stage pipeline used for the report."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

GREY, EDGE, TEXT, ACCENT = '#f1f3f5', '#8a9199', '#1f2933', '#1f6feb'

fig, ax = plt.subplots(figsize=(11, 4.4))
ax.set_xlim(0, 100)
ax.set_ylim(-1.5, 44)
ax.axis('off')

def box(x, y, w, h, title, body, accent=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2',
                                fc='#e8f0fe' if accent else GREY, ec=ACCENT if accent else EDGE, lw=1.2))
    ax.text(x + w / 2, y + h - 1.6, title, ha='center', va='top', fontsize=9, weight='bold', color=TEXT)
    ax.text(x + w / 2, y + h - 5.0, body, ha='center', va='top', fontsize=7.6, color=TEXT, linespacing=1.35)
    return (x, y, w, h)

def arrow(x0, y0, x1, y1, dashed=False):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.2, ls='--' if dashed else '-', shrinkA=0, shrinkB=0))

inp = box(0.5, 14, 14, 17, 'Inputs', '165 transaction\nfeatures per node\n\nedge list\n(234,355 fund flows)')
st = box(18.5, 25, 21.5, 11, 'Structural (12)', 'degree, DAG depth,\nancestors, peel-chain runs')
nb = box(18.5, 9, 21.5, 11, 'One-hop aggregates (82)', 'mean/max of 20 features over\nvisible funders and spenders')
ma = box(43, 12, 16, 21, 'Model A', 'LightGBM, defaults\n259 features\n\nout-of-fold scores,\n5 folds grouped\nby time step')
rs = box(63, 12, 15, 21, 'Neighbour risk', 'mean/max of model-A\nscores over in, out\nand all neighbours (6)\n\n+ two-hop scores (4)\nfor exp11b only')
mb = box(82, 12, 17.5, 21, 'Model B', 'LightGBM extra_trees\n259 features + risk\n\nexp11a / 11b / 11c\neach averaged over\n5 seeds (42-46)')
f1 = box(63, -1, 15, 9.5, 'exp11c  (final 1)', 'public AUC 0.96925', accent=True)
f2 = box(82, -1, 17.5, 9.5, 'exp11g  (final 2)', 'rank average of 11a/b/c\npublic AUC 0.96873', accent=True)

arrow(15.1, 26, 18.1, 30.5)
arrow(15.1, 19, 18.1, 14.5)
arrow(40.4, 30.5, 42.6, 26)
arrow(40.4, 14.5, 42.6, 19)
arrow(59.4, 22.5, 62.6, 22.5)
arrow(78.4, 22.5, 81.6, 22.5)
arrow(86, 11.6, 74, 8.9)
arrow(90.75, 11.6, 90.75, 8.9)
# model B also uses the 259 own features (dashed line over the top)
ax.plot([51, 51, 90.75], [33.4, 40, 40], color=EDGE, lw=1.1, ls='--')
arrow(90.75, 40, 90.75, 33.4, dashed=True)
ax.text(71, 40.8, 'the same 259 features also go to model B, alongside the risk scores', ha='center', fontsize=7.6, color='#52606d')
ax.text(29.25, 6.2, 'visibility mask: only labelled-train and test\ntransactions count as neighbours',
        ha='center', va='top', fontsize=7.2, color='#52606d', style='italic')

out = Path(__file__).with_name('pipeline.png')
fig.savefig(out, dpi=220, bbox_inches='tight', facecolor='white')
print('saved', out)
