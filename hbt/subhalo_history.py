#!/usr/bin/python3
from astropy.cosmology import FlatLambdaCDM
from glob import glob
from itertools import count
from matplotlib import pyplot as plt, ticker, rcParams
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
import numpy as np
import os
import sys
from time import time

from plottools.plotutils import colorscale, savefig, update_rcParams
update_rcParams()
rcParams['text.latex.preamble'].append(r'\usepackage{color}')

from HBTReader import HBTReader

# local
from core import hbt_tools
from core.simulation import Simulation
from core.subhalo import Subhalos, Track
#from core.subhalo_new import Subhalos, Track

adjust_kwargs = dict(
    left=0.10, right=0.95, bottom=0.05, top=0.98, wspace=0.3, hspace=0.1)

def main():

    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation)

    to = time()
    reader = HBTReader(sim.path)
    print('Loaded reader in {0:.1f} seconds'.format(time()-to))
    to = time()
    subs = Subhalos(reader.LoadSubhalos(-1), sim)
    #subs.sort(order='Mbound')
    print('Loaded subhalos in {0:.2f} minutes'.format((time()-to)/60))

    print('In total there are {0} central and {1} satellite subhalos'.format(
        subs.centrals.size, subs.satellites.size))

    centrals = Subhalos(subs.centrals, sim)
    satellites = Subhalos(subs.satellites, sim)

    # sort host halos by mass
    print('Sorting by mass...')
    rank = {'Mbound': np.argsort(-centrals.Mbound)}
    # should not count on them being sorted by mass
    ids_cent = subs.centrals['TrackId']

    n = 20
    print('Plotting centrals...')
    to = time()
    plot_centrals(
        sim, reader, subs, subs.centrals[rank['Mbound'][:n]],
        title='{1}: {0} most massive central subhalos'.format(
            n, sim.formatted_name))
    print('Finished plot_centrals in {0:.2f} minutes'.format((time()-to)/60))
    print('Plotting halos...')
    to = time()
    # total, gas, halo, stars
    for massindex in range(-1, 6):
        plot_halos(
            sim, reader, subs, subs.centrals['TrackId'][rank['Mbound']],
            massindex=massindex)
        break
    print('Finished plot_halos in {0:.2f} minutes'.format((time()-to)/60))
    return


def plot_centrals(sim, reader, subcat, centrals, title='Central subhalos'):
    colspan = 7
    fig = plt.figure(figsize=(2*colspan+1,7))
    nrows = title_axis(title)
    axes = [plt.subplot2grid(
                (nrows,2*colspan+1), (1,0), colspan=colspan, rowspan=nrows-1),
            plt.subplot2grid(
                (nrows,2*colspan+1), (1,colspan), colspan=colspan,
                rowspan=nrows-1)]
    cscale = colorscale(10+np.log10(centrals['Mbound']))
    for i, color in enumerate(cscale[0]):
        plot_track(
            sim, reader, subcat, centrals, i, axes, color=color,
            label_host=False, verbose=(i%200==0), show_history=False)
    cax = plt.subplot2grid((nrows,2*colspan+1), (1,2*colspan), rowspan=nrows-1)
    cbar = plt.colorbar(cscale[1], cax=cax)
    cbar.ax.tick_params(labelsize=12, direction='out', which='both', pad=14)
    cbar.set_label('log Mbound (z=0)', fontsize=12)
    setup_track_axes(axes, sim.cosmology)
    output = os.path.join(sim.plot_path, 'track_centrals.pdf')
    savefig(output, fig=fig)
    return


def plot_halo(sim, reader, subcat, hostid, axes, massindex=-1, output='',
              nsub=10, fig=None, show_label=False):
    cat = subcat.catalog
    # find all subhalos in the same halo
    central = (cat['TrackId'] == hostid)
    central_hostid = cat['HostHaloId'][central]
    halo = (cat['HostHaloId'] == central_hostid)
    print_halo(cat[halo])
    ranked = np.argsort(-cat[halo]['Mbound'])
    # central subhalo
    plot_track(sim, reader, subcat, cat[halo], ranked[0], axes, color='k',
               lw=3, massindex=massindex, show_label=show_label)
    # most massive satellite subhalos
    for i, n in enumerate(ranked[1:nsub+1]):
        plot_track(
            sim, reader, subcat, cat[halo], n, axes, massindex=massindex,
            lw=1, color='C{0}'.format(i), show_label=show_label)
    if show_label:
        axes[1].legend(
            fontsize=12, bbox_to_anchor=(0.96,0.54), loc='upper right')
    if output:
        plt.subplots_adjust(**adjust_kwargs)
        savefig(output, fig=fig, close=False, tight=False)
    return


def plot_halos(sim, reader, subcat, ids_central, massindex=-1, ncl=3):
    """Plot the evolution of a few massive subhalos in the most massive
    halos"""
    mname = sim.masslabel(index=massindex, latex=False)
    output = 'track_{0}.pdf'.format(mname)
    output = os.path.join(sim.plot_path, output)
    fig = plt.figure(figsize=(14,5*(ncl+0.5)))
    title = '{0}: {1} most massive halos\n'.format(sim.formatted_name, ncl)
    #title += 'Mass: {0}\n'.format(mname)
    #title += 'Bold: central subhalo\n'
    #title += r'Plus sign: $t_\mathrm{cent}$ Cross: $t_\mathrm{infall}$' + '\n'
    title += r'Empty: $t_\mathrm{cent}$ Filled: $t_\mathrm{infall}$' + '\n'
    title += 'Legend format: TrackID: log{0} (Depth/HostHaloID)'.format(mname)
    nrows = title_axis(title, rows=10)
    gridsize = (nrows, 2)
    rowspan = nrows // ncl
    axloc = np.array(
        [[(1+i*rowspan,j) for j in range(2)] for i in range(ncl)])
    axes = [[plt.subplot2grid(gridsize, (1+i*rowspan,j), rowspan=rowspan)
             for j in range(2)] for i in range(ncl)]
    for i, row, hostid in zip(count(), axes, ids_central):
        plot_halo(sim, reader, subcat, hostid, row, output=output,
                  massindex=massindex, fig=fig)
        setup_track_axes(
            row, sim.cosmology, is_last_row=(i==len(axes)-1),
            masslabel=sim.masslabel(index=massindex, latex=True))
    plt.subplots_adjust(**adjust_kwargs)
    savefig(output, fig=fig, tight=False)
    return


def plot_track(sim, reader, subhalos, cat, index, axes, show_label=True,
               color='k', massindex=-1, show_history=True, label_host=True,
               verbose=True, **kwargs):
    trackid = cat['TrackId'][index]
    ti = time()
    track = Track(reader.GetTrack(trackid), sim)
    Mt = track.mass(index=massindex)
    Mo = Mt[-1]
    if verbose:
        print('Loaded track #{2} (TrackID {0}) in {1:.2f} minutes'.format(
            trackid, (time()-ti)/60, index))
    if show_label:
        label = '{0}: {1:.2f} ({2}/{3})'.format(
            trackid, np.log10(Mt[-1]), track.track['Depth'][-1],
            track.track['HostHaloId'][-1])
    else:
        label = '_none_'
    t = track.lookback_time()
    axes[0].plot(t, Mt, label=label, color=color, **kwargs)
    axes[1].plot(t, Mt/Mo, label=label, color=color, **kwargs)

    if show_history:
        # las moment it was a central
        icent = track.last_central_snapshot_index
        # infall
        iinf = track.infall('index')
        for ax, m in zip(axes, [Mt,Mt/Mo]):
            ax.plot(t[icent], m[icent], 'wo', mec=color, ms=12, mew=1.5)
            ax.plot(t[iinf], m[iinf], 'o', ms=7, color=color, mew=1.5)

    # the main host halo - just plot when looking at the central
    # subhalo
    if track.track['Rank'][-1] == 0 and show_history:
        host = track.host(-1, return_value='track')
        host = Track(host, sim)
        th = host.lookback_time()
        host_kwargs = dict(dashes=(6,6), color='k', lw=1.5)
        #axes[0].plot(th, 1e10*host.track[mkey], **host_kwargs)
        #axes[1].plot(th, host.track[mkey]/host.track[mkey][-1], **host_kwargs)
        # some information on the host halo
        if label_host:
            text = r'log {1}$_\mathrm{{,host}} = {0:.2f}$'.format(
                np.log10(host.mass(index=massindex)[-1]),
                sim.masslabel(index=massindex, latex=True))
            text += '\nHost ID = {0}'.format(host.trackid)
            txt = axes[0].text(
                0.04, 0.97, text, ha='left', va='top', fontsize=16,
                transform=axes[0].transAxes)
            txt.set_bbox(dict(facecolor='w', edgecolor='w', alpha=0.5))
    return


def print_halo(halo, mmin=10):
    print()
    print('HostHaloId {0}'.format(halo['TrackId']))
    print(halo['Mbound'], (halo['Rank'] == 0).sum())
    print('has a total mass {0:.2e} Msun/h'.format(
        1e10*halo['Mbound'][halo['Rank'] == 0][0]))
    print('and {0} subhalos (including the central),'.format(
        halo['Rank'].size))
    print('with')
    print('    {0} disrupted subhalos'.format((halo['Nbound'] == 1).sum()))
    for mmin in np.logspace(-2, 4, 7):
        print('    {0} having Mbound > {1:.2e} Msun/h'.format(
            (halo['Mbound'] > mmin).sum(), 1e10*mmin))
    return


def summarize_array(array, name, **kwargs):
    print('{0}: (min,max)=({1},{2}); size={3}'.format(
        name, array.min(), array.max(), array.size), **kwargs)
    return


def title_axis(title, rows=10):
    """Title axis wrapper"""
    tax = plt.subplot2grid((rows,1), (0,0))
    tax.axis('off')
    tax.text(0.5, 0.5, title, ha='center', va='center')
    tax.set_xlim(0, 1)
    tax.set_ylim(0, 1)
    return rows


def setup_track_axes(axes, lookback_cosmo=False, textcolor='0.3',
                     masslabel=r'$M_\mathrm{total}$', is_last_row=True):
    if isinstance(lookback_cosmo, FlatLambdaCDM):
        x = 't'
        xlabel = 'lookback time (Gyr)'
        xlim = (14.5, -0.5)
        tickspace = 4
        zmark = [0.1, 0.5, 1, 5]
        tmark = [lookback_cosmo.lookback_time(z).value for z in zmark]
    else:
        x = 'z'
        xlabel = 'Redshift'
        xlim = (16, -0.5)
        tickspace = 5
    axes[0].set_ylabel(r'{1}$({0})$'.format(x, masslabel))
    axes[1].set_ylabel(r'{1}$({0})/${1}$_{{,0}}$'.format(x, masslabel))
    for ax in axes:
        ax.set_yscale('log')
        ax.set_xlim(*xlim)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tickspace))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ytext = get_ytext(ax)
        if is_last_row:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xticklabels([])
        if lookback_cosmo:
            for z, t in zip(zmark, tmark):
                ax.axvline(t, dashes=(4,4), lw=1, color=textcolor)
                # in order to do this, need to know plot limits...
                ax.annotate(
                    'z={0:.1f}'.format(z), xy=(t-0.1,ytext), ha='left',
                    va='center', fontsize=12, color=textcolor)
    return


def get_ytext(ax, height=0.05):
    ylim = ax.get_ylim()
    if ax.get_yscale() == 'log':
        logy = np.log10(np.array(ylim))
        ytext = 10**(logy[0] + height*(logy[1]-logy[0]))
    else:
        ytext = ylim[0] + height*(ylim[1]-ylim[0])
    return ytext


main()




