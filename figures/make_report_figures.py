"""Draws figures/drift_by_step.png and figures/folds_vs_public.png for the report."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).parent
ROOT = HERE.parent
GREY, ACCENT, TEXT, MUTED = '#b8bec6', '#1f6feb', '#1f2933', '#52606d'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': '#8a9199', 'axes.labelcolor': TEXT,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})

# 1. illicit share and labelled count per training time step
tr = pd.read_csv(ROOT / 'data' / 'train.csv', usecols=['time_step', 'label'])
lab = tr[tr.label.notna()]
by = lab.groupby('time_step').label.agg(['size', 'mean'])
overall = lab.label.mean()

fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.bar(by.index, by['size'], color=GREY, width=0.75, label='labelled transactions (left axis)')
ax.set_ylabel('labelled transactions')
ax.set_xlabel('time step (training period 1-35; test period 36-49 not shown)')
ax2 = ax.twinx()
ax2.plot(by.index, by['mean'] * 100, color=ACCENT, marker='o', ms=3, lw=1.6, label='illicit share (right axis)')
ax2.axhline(overall * 100, color=MUTED, ls='--', lw=1, label=f'overall illicit share {overall:.1%}')
ax2.set_ylabel('illicit share of labelled (%)', color=ACCENT)
ax2.spines['right'].set_visible(True)
ax2.set_ylim(0, by['mean'].max() * 100 * 1.15)
ax.set_xlim(0.3, 35.7)
ax.set_ylim(0, by['size'].max() * 1.35)
lo, hi = by['mean'].idxmin(), by['mean'].idxmax()
ax.set_title(f"Illicit share swings from {by['mean'].min():.1%} (step {lo}) to {by['mean'].max():.1%} (step {hi}) across training steps",
             fontsize=9.5, color=TEXT, loc='left')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper left', frameon=False, fontsize=7.5, ncol=3)
fig.savefig(HERE / 'drift_by_step.png', dpi=220, bbox_inches='tight', facecolor='white')
plt.close(fig)

# 2. validation folds vs public AUC for every submitted model
log = pd.read_csv(ROOT / 'experiment_log.csv')
s = log[log.public_lb.notna()].copy()
s['recent'] = s[['auc_F2', 'auc_F3', 'auc_F4']].mean(axis=1)   # exp01 predates F4: mean of F2/F3
finals = {'exp11c', 'exp11g'}

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, col, name in [(axes[0], 'auc_F1', 'F1 AUC (validate steps 21-27)'), (axes[1], 'recent', 'mean F2-F4 AUC (recent folds)')]:
    rho = spearmanr(s[col], s.public_lb).statistic
    for _, r in s.iterrows():
        fin = r.exp_id in finals
        ax.scatter(r[col], r.public_lb, s=34 if fin else 22, color=ACCENT if fin else GREY,
                   edgecolor=ACCENT if fin else '#8a9199', zorder=3)
    labels = ([('exp01', 6, -3), ('exp03', -12, -13), ('exp09b', 5, -10), ('exp11c', 7, 2)] if col == 'auc_F1'
              else [('exp01', 6, -3), ('exp03', 6, -3), ('exp11c', -42, 3)])
    for e, dx, dy in labels:
        r = s[s.exp_id == e].iloc[0]
        ax.annotate(e, (r[col], r.public_lb), xytext=(dx, dy), textcoords='offset points', fontsize=7, color=MUTED)
    ax.margins(x=0.08, y=0.1)
    ax.set_xlabel(name)
    ax.set_title(f'Spearman rank correlation {rho:+.2f}', fontsize=8.5, color=TEXT, loc='left')
axes[0].set_ylabel('public leaderboard AUC')
fig.suptitle('Recent folds track the public leaderboard; F1 does not (10 submissions; blue = the two finals)',
             fontsize=9.5, color=TEXT, x=0.01, ha='left')
fig.tight_layout()
fig.savefig(HERE / 'folds_vs_public.png', dpi=220, bbox_inches='tight', facecolor='white')
print('saved', HERE / 'drift_by_step.png', 'and', HERE / 'folds_vs_public.png')
print(s[['exp_id', 'auc_F1', 'recent', 'public_lb']].round(4).to_string(index=False))
