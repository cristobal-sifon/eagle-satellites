from astropy.cosmology import FlatLambdaCDM
from glob import glob
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
from core import Simulation
import hbt_tools


def main():

    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation)

    to = time()
    reader = HBTReader(sim.path)
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
    print('Sorting by mass...')
    rank = {'Mbound': np.argsort(-subs['Mbound'][cent])}
    #ids_cent = np.array([subs['HostHaloId'][cent][i] for i in rank['Mbound']])
    ids_cent = subs['HostHaloId'][cent][rank['Mbound']]

    n = 20
    print('Plotting centrals...')
    to = time()
    plot_centrals(
        sim, reader, subs[cent][rank['Mbound'][:n]],
        title='{1}: {0} most massive central subhalos'.format(
            n, sim.formatted_name))
    print('Finished plot_centrals in {0:.2f} minutes'.format((time()-to)/60))
    print('Plotting halos...')
    to = time()
    plot_halos(sim, reader, subs, ids_cent)
    print('Finished plot_halos in {0:.2f} minutes'.format((time()-to)/60))
    return


def plot_centrals(sim, reader, centrals, title='Central subhalos'):
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
        #to = time()
        plot_track(
            reader, centrals, i, sim.cosmology, axes, color=color,
            verbose=(i%200==0))
        #print('{0:4d}) {1:6.2f} s'.format(i, time()-to))
    cax = plt.subplot2grid((nrows,2*colspan+1), (1,2*colspan), rowspan=nrows-1)
    cbar = plt.colorbar(cscale[1], cax=cax)
    cbar.ax.tick_params(labelsize=12, direction='out', which='both', pad=14)
    cbar.set_label('log Mbound (z=0)', fontsize=12)
    setup_track_axes(axes, sim.cosmology)
    output = os.path.join(sim.plot_path, 'track_centrals.pdf')
    savefig(output, fig=fig)
    return


def plot_halo(reader, cat, hostid, cosmo, axes, output='', nsub=5, fig=None):
    halo = (cat['HostHaloId'] == hostid)
    print_halo(cat[halo])
    ranked = np.argsort(-cat[halo]['Mbound'])
    # central subhalo
    plot_track(reader, cat[halo], ranked[0], cosmo, axes, color='k', lw=3)
    # most massive satellite subhalos
    for i in ranked[1:nsub+1]:
        plot_track(reader, cat[halo], i, cosmo, axes, lw=1)
        if output:
            for ax in axes:
                ax.legend(fontsize=12, loc='upper left')
            savefig(output, fig=fig, close=False)
    return


def plot_halos(sim, reader, cat, ids_central, ncl=3):
    """plot the evolution of a few massive subhalos in the most massive
    halos"""
    output = os.path.join(sim.plot_path, 'track_masses.pdf')
    #fig, axes = plt.subplots(figsize=(12,5*ncl), ncols=2, nrows=ncl)
    fig = plt.figure(figsize=(12,5*(ncl+0.5)))
    title = '{0}: {1} most massive halos\n'.format(sim.formatted_name, ncl)
    #title += r'$\bf{{Bold:}}$ Central --' \
             #r' $\textcolor{{red}}{{Color:}}$ Satellite'
    nrows = title_axis(title, rows=10)
    #gridsize = ((nrows+1)*ncl, 2)
    gridsize = (nrows, 2)
    #print('gridsize =', gridsize)
    rowspan = nrows // ncl
    axloc = np.array(
        [[(1+i*rowspan,j) for j in range(2)] for i in range(ncl)])
    #print('axloc =', axloc)
    axes = [[plt.subplot2grid(gridsize, (1+i*rowspan,j), rowspan=rowspan)
             for j in range(2)] for i in range(ncl)]
    for row, hostid in zip(axes, ids_central):
        #row[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%s'))
        plot_halo(reader, cat, hostid, sim.cosmology, row, output=output,
                  fig=fig)
        setup_track_axes(row, sim.cosmology)
    savefig(output, fig=fig)
    return


def plot_track(reader, cat, index, cosmo, axes, show_label=True,
               mtype='', verbose=True, **kwargs):
    trackid = cat['TrackId'][index]
    ti = time()
    track = reader.GetTrack(trackid)
    if verbose:
        print('Loaded track #{2} (TrackID {0}) in {1:.2f} minutes'.format(
            trackid, (time()-ti)/60, index))
    if show_label:
        label = '{0}: {1:.2f}'.format(
            trackid, np.log10(1e10*cat['Mbound'][index]))
    else:
        label = '_none_'
    z = 1/track['ScaleFactor'] - 1
    t = cosmo.lookback_time(z)
    axes[0].plot(t, 1e10*track['Mbound'], label=label, **kwargs)
    axes[1].plot(
        t, track['Mbound']/track['Mbound'][-1], label=label, **kwargs)
    return


def print_halo(halo, mmin=10):
    print()
    print('HostHaloId {0}'.format(halo['TrackId'][0]))
    print('has a total mass {0:.2e} Msun/h'.format(
        1e10*halo[halo['Rank'] == 0]['Mbound'][0]))
    print('and {0} subhalos (including the central),'.format(
        halo['Rank'].size))
    print('with')
    print('    {0} disrupted subhalos'.format((halo['Nbound'] == 1).sum()))
    for mmin in np.logspace(-2, 4, 7):
        print('    {0} having Mbound > {1:.2e} Msun/h'.format(
            (halo['Mbound'] > mmin).sum(), 1e10*mmin))
    return


def title_axis(title, rows=10):
    """Title axis wrapper"""
    tax = plt.subplot2grid((rows,1), (0,0))
    tax.axis('off')
    tax.text(0.5, 0.5, title, ha='center', va='center')
    tax.set_xlim(0, 1)
    tax.set_ylim(0, 1)
    return rows


def setup_track_axes(axes, lookback_cosmo=False, textcolor='0.3'):
    if isinstance(lookback_cosmo, FlatLambdaCDM):
        x = 't'
        xlabel = 'lookback time (Gyr)'
        xlim = (14, -0.5)
        zmark = [0.1, 0.5, 1, 5]
        tmark = [lookback_cosmo.lookback_time(z).value for z in zmark]
    else:
        x = 'z'
        xlabel = 'Redshift'
        xlim = (16, -0.5)
    axes[0].set_ylabel(r'$M({0})$'.format(x))
    axes[1].set_ylabel(r'$M({0})/M_0$'.format(x))
    for ax in axes:
        ax.set_yscale('log')
        ax.set_xlim(*xlim)
        ytext = get_ytext(ax)
        #if ax.is_last_row():
        ax.set_xlabel(xlabel)
        #else:
            #ax.set_xticklabels([])
        if lookback_cosmo:
            for z, t in zip(zmark, tmark):
                ax.axvline(t, dashes=(4,4), lw=1, color=textcolor)
                # in order to do this, need to know plot limits...
                ax.annotate(
                    'z={0:.1f}'.format(z), xy=(t-0.1,ytext), ha='left',
                    va='center', fontsize=12, color=textcolor)
    return


def get_ytext(ax, height=0.06):
    ylim = ax.get_ylim()
    if ax.get_yscale() == 'log':
        logy = np.log10(np.array(ylim))
        print('ylim =', ylim)
        logheight = logy[0] + height*(logy[1] - logy[0])
        print('logheight =', logheight)
        ytext = 10**logheight
        print('ytext =', ytext)
    else:
        ytext = ylim[0] + height*(ylim[1]-ylim[0])
    return ytext


main()



#print(track['ScaleFactor'])
#print(1/track['ScaleFactor'] - 1)



