from matplotlib import pyplot as plt, ticker
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
import numpy as np
from time import time

from plottools.plotutils import savefig, update_rcParams
update_rcParams()

from HBTReader import HBTReader


path_hbt = '/cosma/home/jvbq85/data/HBT/data/eagle/L0100N1504/subcat'

to = time()
reader = HBTReader(path_hbt)
print('Loaded reader in {0:.1f} seconds'.format(time()-to))

to = time()
subs = reader.LoadSubhalos(-1)
print('Loaded subhalos in {0:.2f} minutes'.format((time()-to)/60))

print(type(subs))
print(subs.dtype)

cent = (subs['Rank'] == 0)
sub = (subs['Rank'] > 0)

for key in ('Mbound', 'LastMaxMass', 'Nbound'):
    print('{0}: {1} {2} {3}'.format(
        key, subs[key][sub].min(), subs[key][sub].max(),
        np.percentile(subs[key][sub], [1,5,25,50,75,95,99])))

# this takes forever
"""
to = time()
last_isolation_z = np.array(
    [reader.GetScaleFactor(snap)
     for snap in subs['SnapshotIndexOfLastIsolation']])
print('Recorded last isolation redshift in {0:.2f} minutes'.format(
        (time()-to)/60))
"""
#print(reader.GetScaleFactorDict())
# snap = 0 --> z = 0.1
# snap = 365 --> z = 1.0

# some plots to understand what's going on
def make_hist_sub(column, ax, bins=50, log=False, log_hist=True):
    print('plotting {0} ...'.format(column))
    if log:
        mask = (subs[column] > 0)
        col = np.log10(subs[column])
        xlabel = 'log({0})'.format(column)
    else:
        col = subs[column]
        xlabel = column
        mask = np.ones(col.size, dtype=bool)
    if log_hist:
        ylabel = '1+N'
    else:
        ylabel = 'N'
    ax.hist(col[mask], bins, histtype='step', lw=2, label='all', log=log_hist,
            bottom=1*log_hist)
    ax.hist(col[mask & (subs['Rank'] == 0)], bins, histtype='step', lw=2,
            log=log_hist, bottom=1*log_hist, label='centrals')
    ax.hist(col[mask & (subs['Rank'] > 0)], bins, histtype='step', lw=2,
            log=log_hist, bottom=1*log_hist, label='subhalos')
    ax.legend(fontsize=13, loc='lower center')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
columns = ('Mbound', 'Nbound', 'LastMaxMass', 'SnapshotIndexOfLastMaxMass',
           'SnapshotIndexOfLastIsolation')
log = (True, True, True, False, False)
ncol = len(columns)
fig, axes = plt.subplots(figsize=(5*ncol,4), ncols=ncol)
for ax, column, log_i in zip(axes, columns, log):
    make_hist_sub(column, ax, log=log_i)
savefig('plots/hist_subhalos.pdf', fig=fig)

bins = {'Mbound': np.logspace(-4.3, 4, 100),
        'Nbound': np.logspace(-0.3, 8, 100),
        'SnapshotIndexOfLastIsolation': np.arange(0, reader.MaxSnap, 2),
        'redshift': np.logspace(-2, 0.3, 50)}
fig, axes = plt.subplots(figsize=(14,6), ncols=2)
# Mbound
ax = axes[0]
hist2d, xe, ye = np.histogram2d(
    subs['SnapshotIndexOfLastIsolation'][sub], subs['Mbound'][sub],
    (bins['SnapshotIndexOfLastIsolation'],bins['Mbound']))
extent = (bins['SnapshotIndexOfLastIsolation'][0],
          bins['SnapshotIndexOfLastIsolation'][-1],
          bins['Mbound'][0], bins['Mbound'][-1])
#img = ax.imshow(
    #np.log10(hist2d.T), origin='lower', aspect='auto', extent=extent)
X, Y = np.meshgrid(xe, ye)
img = ax.pcolor(X, Y, hist2d.T, norm=LogNorm())
cbar = plt.colorbar(img, ax=ax, format=LogFormatterMathtext())
cbar.set_label('Number of subhalos')
#cbar.set_yticks([])
ax.set_ylabel('Mbound')
# Nbound
ax = axes[1]
hist2d, xe, ye = np.histogram2d(
    subs['SnapshotIndexOfLastIsolation'][sub], subs['Nbound'][sub],
    (bins['SnapshotIndexOfLastIsolation'],bins['Nbound']))
extent = (bins['SnapshotIndexOfLastIsolation'][0],
          bins['SnapshotIndexOfLastIsolation'][-1],
          bins['Nbound'][0], bins['Nbound'][-1])
#img = ax.imshow(
    #np.log10(hist2d.T), origin='lower', aspect='auto', extent=extent)
X, Y = np.meshgrid(xe, ye)
img = plt.pcolor(X, Y, hist2d.T, norm=LogNorm())
cbar = plt.colorbar(img, ax=ax, format=LogFormatterMathtext())
cbar.set_label('Number of subhalos')
ax.set_ylabel('Nbound')
# plot formatting
snaps = np.arange(0, 365, 60)
scale_factor = np.array([reader.GetScaleFactor(snap) for snap in snaps])
redshifts = 1/scale_factor - 1
redshifts = ['{0:.2f}'.format(z) for z in redshifts]
for ax in axes:
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(ticker.FixedLocator(snaps))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(redshifts))
    ax.set_xlabel('Last isolation redshift')
fig.tight_layout()
output = 'plots/LastIsolation_Mbound.pdf'
savefig(output, fig=fig)
