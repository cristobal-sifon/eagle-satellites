from glob import glob
from matplotlib import pyplot as plt, ticker
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
import numpy as np
import os
import sys
from time import time

from plottools.plotutils import colorscale, savefig, update_rcParams
update_rcParams()

import HBTReader
print(HBTReader.__file__)
from HBTReader import HBTReader


#path_hbt = '/cosma/home/jvbq85/data/HBT/data/eagle/L0100N1504/subcat'
path_hbt = '/cosma/home/jvbq85/data/HBT/data/apostle/V1_LR/subcat'

plot_path = os.path.join('plots', '_'.join(path_hbt.split('/')[-3:-1]))
if not os.path.isdir(plot_path):
    os.makedirs(plot_path)

def main():

    to = time()
    reader = HBTReader(path_hbt)
    print('Loaded reader in {0:.1f} seconds'.format(time()-to))

    to = time()
    subs = reader.LoadSubhalos(-1)
    print('Loaded subhalos in {0:.2f} minutes'.format((time()-to)/60))

    cent = (subs['Rank'] == 0)
    sub = (subs['Rank'] > 0)
    print('In total there are {0} central and {1} satellite subhalos'.format(
        cent.sum(), sub.sum()))

    submass = subs['Mbound'][sub]
    subdm = subs['MboundType'][:,0][sub]
    submstar = subs['MboundType'][:,1][sub]

    # sort host halos by mass
    rank = {'Mbound': np.argsort(-subs['Mbound'][cent])}
    ids_cent = np.array([subs['HostHaloId'][cent][i] for i in rank['Mbound']])

    plot_centrals(reader, subs[cent][rank['Mbound'][:20]])
    plot_clusters(reader, subs, ids_cent)

    return


def plot_centrals(reader, centrals):
    colspan = 6
    fig = plt.figure(figsize=(2*colspan+1,5))
    axes = [plt.subplot2grid((1,2*colspan+1), (0,0), colspan=colspan),
            plt.subplot2grid((1,2*colspan+1), (0,colspan), colspan=colspan)]
    cscale = colorscale(10+np.log10(centrals['Mbound']))
    for i, color in enumerate(cscale[0]):
        plot_track(reader, centrals, i, axes, color=color, verbose=(i%200==0))
    cax = plt.subplot2grid((1,2*colspan+1), (0,2*colspan))
    cbar = plt.colorbar(cscale[1], cax=cax)
    cbar.ax.tick_params(labelsize=12, direction='out', which='both', pad=14)
    cbar.set_label('log Mbound (z=0)', fontsize=12)
    output = os.path.join(plot_path, 'track_centrals.pdf')
    savefig(output, fig=fig)
    return


def plot_cluster(reader, cat, hostid, axes, output='', fig=None):
    cluster = (cat['HostHaloId'] == hostid)
    print_cluster(cat[cluster])
    ranked = np.argsort(-cat[cluster]['Mbound'])
    # central subhalo
    plot_track(reader, cat[cluster], ranked[0], axes, color='k', lw=3)
    # most massive satellite subhalos
    for i in ranked[1:6]:
        plot_track(reader, cat[cluster], i, axes, lw=1)
        if output:
            for ax in axes:
                ax.legend(fontsize=12, loc='upper left')
            savefig(output, fig=fig, close=False)
    return


def plot_clusters(reader, cat, ids_central):
    """plot the evolution of a few massive subhalos in the most massive
    clusters"""
    output = os.path.join(plot_path, 'track_masses.pdf')
    ncl = 3
    fig, axes = plt.subplots(figsize=(12,5*ncl), ncols=2, nrows=ncl)
    for row, hostid in zip(axes, ids_central):
        #row[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%s'))
        plot_cluster(reader, cat, hostid, row, output=output, fig=fig)
    savefig(output, fig=fig)
    return


def plot_track(reader, cat, index, axes, show_label=True, verbose=True,
               **kwargs):
    trackid = cat['TrackId'][index]
    ti = time()
    track = reader.GetTrack(trackid)
    if verbose:
        print('Loaded TrackID {0} in {1:.2f} minutes'.format(
            trackid, (time()-ti)/60))
    if show_label:
        label = '{0}: {1:.2f}'.format(
            trackid, np.log10(1e10*cat['Mbound'][index]))
    else:
        label = '_none_'
    z = 1/track['ScaleFactor'] - 1
    axes[0].plot(z, 1e10*track['Mbound'], label=label, **kwargs)
    axes[1].plot(
        z, track['Mbound']/track['Mbound'][-1], label=label, **kwargs)
    axes[0].set_ylabel(r'$M(z)$')
    axes[1].set_ylabel(r'$M(z)/M_0$')
    for ax in axes:
        ax.set_yscale('log')
        ax.set_xlabel('Redshift')
        ax.set_xlim(16, -0.8)
    return


def print_cluster(cluster, mmin=10):
    print()
    print('The most massive halo has HostHaloId {0}'.format(
        cluster['TrackId'][0]))
    print('and a mass {0}'.format(cluster[cluster['Rank'] == 0]['Mbound'][0]))
    print('and {0} subhalos (including the central),'.format(
        cluster['Rank'].size))
    print('with')
    print('    {0} disrupted subhalos'.format((cluster['Nbound'] == 1).sum()))
    for mmin in np.logspace(-2, 4, 7):
        print('    {0} having Mbound > {1:.2e} Msun/h'.format(
            (cluster['Mbound'] > mmin).sum(), 1e10*mmin))
    return

main()



#print(track['ScaleFactor'])
#print(1/track['ScaleFactor'] - 1)



