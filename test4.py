import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 26
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.rc('axes', titlesize=26, labelsize=26, grid=True)
plt.rc('lines', linewidth=2)
plt.rc('legend', fontsize=26, frameon=False)
plt.rc('xtick', labelsize=26, direction='in')
plt.rc('ytick', labelsize=26, direction='in')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

# --- Example data ---
n = np.array([100, 250, 500, 1000])
nll_mc = np.array([14.75, 15.38, 15.64, 15.71])
ksd_mc = np.array([6.40e-01, 9.14e-02, 1.68e-02, 4.27e-03])
nll_bq = np.array([14.74, 15.43, 15.69, 15.75])
ksd_bq = np.array([4.86e-01, 5.94e-02, 5.77e-03, 2.18e-03])

# true NLL
L = 15
p = 0.4
H_site = -(2*p*np.log(p) + (1-2*p)*np.log(1-2*p))
true_nll = L * H_site

# --- Plot main figure without legend ---
fig, ax1 = plt.subplots(figsize=(10,6))

# left y: KSD²
l1, = ax1.plot(n, ksd_mc, 'k-o', label='KSD$^2$ (MC)')
l2, = ax1.plot(n, ksd_bq, 'b-s', label='KSD$^2$ (DBQ)')
ax1.set_yscale('log')
ax1.set_xlabel('Number of Points')
ax1.set_ylabel('KSD$^2$')
ax1.grid(True, which='major', linestyle='--', linewidth=0.5)

# right y: NLL
ax2 = ax1.twinx()
l3, = ax2.plot(n, nll_mc, 'k--o', label='NLL (MC)')
l4, = ax2.plot(n, nll_bq, 'b--s', label='NLL (BQ)')
l5 = ax2.axhline(true_nll, color='gray', linestyle=':', label='True NLL')
ax2.grid(True, which='major', linestyle='--', linewidth=0.5)
ax2.set_ylabel('NLL')

# remove legend here — we'll make a separate one below
plt.tight_layout()
plt.savefig('ksd_nll_main.pdf')
plt.close(fig)  # Close so it doesn't display inline (optional)

# --- Create separate legend figure ---
handles = [
    Line2D([], [], color='k', linestyle='-',  linewidth=2, label='KSD$^2$ (MC)'),
    Line2D([], [], color='b', linestyle='-',  linewidth=2, label='KSD$^2$ (DBQ)'),
    Line2D([], [], color='k', linestyle='--', linewidth=2, label='NLL (MC)'),
    Line2D([], [], color='b', linestyle='--', linewidth=2, label='NLL (BQ)'),
    Line2D([], [], color='k', linestyle=':',  linewidth=2, label='True NLL'),
]

fig_legend = plt.figure(figsize=(6, 1))
fig_legend.legend(
    handles=handles,
    loc='center',
    ncol=5,
    frameon=False,
    handlelength=2.6,
    columnspacing=1.6
)
fig_legend.tight_layout()
fig_legend.savefig('ksd_nll_legend.pdf', bbox_inches='tight')
plt.close(fig_legend)