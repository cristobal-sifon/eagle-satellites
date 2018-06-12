
from matplotlib import pyplot as plt, ticker
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
def make_hist_sub(column, ax, bins=50, log=):
    print('plotting {0} ...'.format(column))
    ax.hist(subs[column], bins, histtype='step', lw=2, label='all')
    ax.hist(subs[column][subs['Rank'] == 0], bins, histtype='step', lw=2,
            label='centrals')
    ax.hist(subs[column][subs['Rank'] > 0], bins, histtype='step', lw=2,
            label='subhalos')
    ax.legend()
    ax.set_xlabel(column)
columns = ('Mbound', 'Nbound', 'LastMaxMass', 'SnapshotIndexOfLastMaxMass',
           'SnapshotIndexOfLastIsolation')
ncol = len(columns)
fig, axes = plt.subplots(figsize=(5*ncol,4), ncols=ncol)
for ax, column in zip(axes, columns):
    make_hist_sub(column, ax)
savefig('plots/hist_subhalos.pdf', fig=fig)

bins = {'Mbound': np.logspace(0, 2, 50),
        'Nbound': np.logspace(-1, 5, 50),
        'SnapshotIndexOfLastIsolation': np.arange(0, reader.MaxSnap, 2),
        'redshift': np.logspace(-2, 0.3, 50)}
fig, axes = plt.subplots(figsize=(14,6), ncols=2)
# Mbound
ax = axes[0]
hist2d, _, _ = np.histogram2d(
    subs['SnapshotIndexOfLastIsolation'][sub], subs['Mbound'][sub],
    (bins['SnapshotIndexOfLastIsolation'],bins['Mbound']))
extent = (bins['SnapshotIndexOfLastIsolation'][0],
          bins['SnapshotIndexOfLastIsolation'][-1],
          bins['Mbound'][0], bins['Mbound'][-1])
img = ax.imshow(
    np.log10(hist2d.T), origin='lower', aspect='auto', extent=extent)
cbar = plt.colorbar(img, ax=ax)
cbar.set_label('log(Number of subhalos)')
ax.set_ylabel('Mbound')
# Nbound
ax = axes[1]
hist2d, _, _ = np.histogram2d(
    subs['SnapshotIndexOfLastIsolation'][sub], subs['Nbound'][sub],
    (bins['SnapshotIndexOfLastIsolation'],bins['Nbound']))
extent = (bins['SnapshotIndexOfLastIsolation'][0],
          bins['SnapshotIndexOfLastIsolation'][-1],
          bins['Nbound'][0], bins['Nbound'][-1])
img = ax.imshow(
    np.log10(hist2d.T), origin='lower', aspect='auto', extent=extent)
cbar = plt.colorbar(img, ax=ax)
cbar.set_label('log(Number of subhalos)')
ax.set_ylabel('Nbound')
# plot formatting
snaps = np.arange(0, 365, 60)
scale_factor = np.array([reader.GetScaleFactor(snap) for snap in snaps])
redshifts = 1/scale_factor - 1
redshifts = ['{0:.2f}'.format(z) for z in redshifts]
for ax in axes:
    ax.xaxis.set_major_locator(ticker.FixedLocator(snaps))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(redshifts))
    ax.set_xlabel('Last isolation redshift')
output = 'plots/LastIsolation_Mbound.pdf'
savefig(output, fig=fig)
