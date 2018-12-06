#!/usr/bin/python3
from astropy.cosmology import FlatLambdaCDM
from glob import glob
from itertools import count
from matplotlib import pyplot as plt, ticker, rcParams
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.colorbar import Colorbar
import multiprocessing as mp
import numpy as np
import os
import sys
from time import sleep, time

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
    print('Running...')
    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation)

    to = time()
    reader = HBTReader(sim.path)
    print('Loaded reader in {0:.1f} seconds'.format(time()-to))
    to = time()
    subs = Subhalos(reader.LoadSubhalos(-1), sim, -1, as_dataframe=False)
    #subs.sort(order='Mbound')
    print('Loaded subhalos in {0:.2f} minutes'.format((time()-to)/60))

    print('In total there are {0} central and {1} satellite subhalos'.format(
        subs.centrals.size, subs.satellites.size))

    centrals = Subhalos(
        subs.centrals, sim, -1, load_distances=False, load_velocities=False)
    satellites = Subhalos(
        subs.satellites, sim, -1, load_distances=False, load_velocities=False)

    # sort host halos by mass
    print('Sorting by mass...')
    rank = {'Mbound': np.argsort(-centrals.Mbound).values}
    # should not count on them being sorted by mass
    ids_cent = subs.centrals['TrackId']

    do_plot_centrals = False
    if do_plot_centrals:
        n = 10
        print('Plotting centrals...')
        to = time()
        title = '{1}: {0} most massive central subhalos'.format(
            n, sim.formatted_name)
        plot_centrals(
            args, sim, reader, subs, subs.centrals, rank['Mbound'][:n],
            #title=None)
            title=title)
        print('Finished plot_centrals in {0:.2f} minutes'.format(
            (time()-to)/60))

    print()
    print('Plotting halos...')
    to = time()
    # total, gas, halo, stars
    # ['gas', 'DM', 'disk', 'bulge', 'stars', 'BH']
    ti = time()
    plot_halos(
        args, sim, reader, subs, subs.centrals['TrackId'][rank['Mbound'][:3]])
    #print('Plotted mass #{0} in {1:.2f} minutes'.format(
        #massindex, (time()-to)/60))
    print('Finished plot_halos in {0:.2f} minutes'.format((time()-to)/60))
    return


def member_indices(args, subcat, host_ids, nsub=10):
    cat = subcat.catalog
    centrals = (cat['Rank'] == 0)
    halos = np.intersect1d(cat['TrackId'], host_ids)
    to = time()
    #if args.ncores == 1:
    if True:
        idx = []
        for track in host_ids:
            idx.append(subcat.siblings(track, 'index')[:nsub+1])
    else:
        idx = [track for track in host_ids]
        pool = mp.Pool(args.ncores)
        results = \
            [pool.apply_async(subcat.siblings, args=(track,),
                              kwds={'return_value': 'index'})
             for track in host_ids]
        pool.close()
        pool.join()
        for out in results:
            out = out.get()
            idx[out[0]] = out[1]
    print('member indices in {0:.2f} s'.format(time()-to))
    return idx


def plot_centrals(args, sim, reader, subcat, centrals, indices, massindex=-1,
                  title='Central subhalos'):
    # another option is to create more than one gridspec, see
    # https://matplotlib.org/users/tight_layout_guide.html
    # but I've wasted enough time here for now
    gridspec_kw = {'width_ratios':(1,1,0.05)}
    fig, (ax1, ax2, cax) = plt.subplots(
        figsize=(15,6), ncols=3, gridspec_kw=gridspec_kw)
    axes = (ax1, ax2)
    if title:
        fig.suptitle(title)
    cscale, cmap = colorscale(10+np.log10(centrals['Mbound'][indices]))
    read_and_plot_track(
        args, axes, sim, reader, subcat, centrals, indices,
        show_history=False, label_host=False, show_label=False,
        colors=cscale, label_history=False)
    cbar = fig.colorbar(cmap, cax=cax)
    cbar.ax.tick_params(labelsize=20, direction='in', which='both', pad=14)
    clabel = sim.masslabel(mtype='total')
    cbar.set_label(r'log ${0}$ ($z=0$)'.format(clabel))#, fontsize=20)
    setup_track_axes(axes, sim, sim.cosmology)
    output = os.path.join(sim.plot_path, 'track_centrals.pdf')
    savefig(output, fig=fig, rect=[0,0,1,0.9])
    return


def make_halo_plot(args, sim, tracks, massindex=-1, includes_central=True,
                   title=None, label_history=True, suffix=''):
    to = time()
    # this should include multiple clusters
    if not hasattr(tracks[0], '__iter__'):
        tracks = [tracks]
    i0 = int(includes_central)
    hostids = [t[-3] for t in tracks]
    ncl = np.unique(hostids).size
    nsub = [(hostids == i).sum() for i in np.unique(hostids)]
    nsat = [n-includes_central for n in nsub]
    # colors. Centrals are black; loop colors for satellites
    colors = []
    j = 0
    n = 0
    for i in range(len(tracks)):
        #if j == sum(nsub[:i]):
        if i == sum(nsub[:n]):
            j = 0
            n += 1
            if includes_central:
                colors.append('k')
                continue
        colors.append('C{0}'.format(j%10))
        j += 1
    mname = sim.masslabel(index=massindex, latex=False)
    output = 'track_{0}'.format(mname)
    if suffix:
        output += '_{0}'.format(suffix)
    output = os.path.join(sim.plot_path, '{0}.pdf'.format(output))
    #title = '{0}: {1} most massive halos'.format(sim.formatted_name, ncl)
    fig, axes = plt.subplots(figsize=(14,5*(ncl+0.5)), ncols=2, nrows=ncl)
    for i, row in enumerate(axes):
        n = int(np.sum(nsub[:i]))
        # the central track
        if includes_central:
            plot_track(
                row, sim, tracks[n], massindex=massindex, color=colors[n],
                label_history='right')
        j = [n+i0+ii for ii in range(nsub[i]-i0)]
        _ = [plot_track(
                row, sim, tracks[ji], massindex=massindex, color=colors[ji])
             for ji in j]
        setup_track_axes(
            row, sim, sim.cosmology, is_last_row=(i==len(axes)-1),
            masslabel=sim.masslabel(index=massindex, latex=True))
    rect = [0.03, 0.01, 0.99, 0.99]
    if title:
        fig.suptitle(title)
        rect[3] -=  0.12/ncl
    savefig(output, fig=fig, rect=rect, w_pad=1.2)
    print('    in {0:.1f} s'.format(time()-to))
    return


def plot_halos(args, sim, reader, subcat, ids_central, nsub=10,
               title=None):
    """Plot the evolution of a few massive subhalos in the most massive
    halos"""
    to = time()
    ncl = len(ids_central)
    idx = member_indices(args, subcat, ids_central, nsub=nsub)
    idx = np.reshape(idx, -1)
    # this is the expensive step. Need to figure out exactly what
    # comes out but I think it should be a nested list with all the
    # tracks, where the first dimension are the hosts
    tracks = read_tracks(args, sim, reader, subcat.catalog, idx, nsub=nsub)
    # (['gas', 'DM', 'disk', 'bulge', 'stars', 'BH']
    # disk and bulge mass not included in EAGLE nor Apostle
    # saves me about a minute
    for massindex in (-1, 0, 1, 4, 5):
        make_halo_plot(args, sim, tracks, massindex=massindex, title=title)
    return


def plot_track(axes, sim, track_data, massindex=-1,
               show_history=True, label_host=True, show_label=True, color='k',
               label_history=False, verbose=True, **kwargs):
    """
    Make sure `massindex` is consistent with `read_track()`
    """
    trackid, t, Mt, rank, depth, icent, isat, iinf, hostid, th, Mh = track_data
    if hasattr(massindex, '__iter__'):
        pass
    else:
        Mt = Mt[massindex]
        Mh = Mh[massindex]
    Mo = Mt[-1]
    if show_label:
        label = '{0}: {1:.2f} ({2}/{3})'.format(
            trackid, np.log10(Mt[-1]), depth[-1], hostid)
    else:
        label = '_none_'
    axes[0].plot(t, Mt, label=label, color=color, **kwargs)
    axes[1].plot(t, Mt/Mo, color=color, **kwargs)

    if show_history:
        for ax, m in zip(axes, [Mt,Mt/Mo]):
            if icent is not None:
                ax.plot(t[icent], m[icent], 'ws', mec=color, ms=12, mew=1.5,
                        label=r'$t_\mathrm{cen}$')
            if isat is not None:
                ax.plot(t[isat], m[isat], 'wo', mec=color, ms=12, mew=1.5,
                        label=r'$t_\mathrm{sat}$')
            else:
                ax.plot([], [], 'wo', mec=color, ms=12, mew=1.5,
                        label=r'$t_\mathrm{sat}$')
            ax.plot(t[iinf], m[iinf], 'o', ms=7, color=color, mew=1.5,
                    label=r'$t_\mathrm{infall}$')
        if label_history:
            lax = axes[['left', 'right'].index(label_history)]
            lax.legend(loc='lower right', bbox_to_anchor=(0.97,0.08))
    # the main host halo - just plot when looking at the central
    # subhalo
    if rank[-1] == 0 and show_history:
        #host_kwargs = dict(dashes=(6,6), color='k', lw=1.5)
        #axes[0].plot(th, Mh, **host_kwargs)
        #axes[1].plot(th, Mh/Mh[-1], **host_kwargs)
        # some information on the host halo
        if label_host:
            text = r'log ${1}^\mathrm{{host}} = {0:.2f}$'.format(
                np.log10(Mh[-1]),
                sim.masslabel(index=massindex, latex=True))
            text += '\nHost ID = {0}'.format(hostid)
            txt = axes[0].text(
                0.04, 0.97, text, ha='left', va='top', fontsize=16,
                transform=axes[0].transAxes)
            txt.set_bbox(dict(facecolor='w', edgecolor='w', alpha=0.5))
    return


def read_and_plot_track(args, axes, sim, reader, subhalos, cat, indices,
                        massindex=-1, show_history=True, label_host=True,
                        show_label=True, colors=None, color='k',
                        label_history=False, verbose=True, **kwargs):
    if not hasattr(indices, '__iter__'):
        indices = [indices]
    read_kwargs = {'massindex': massindex, 'load_history': show_history,
                   'verbose': verbose}
    if colors is None:
        colors = [color]*len(indices)
    if args.ncores == 1:
        track_data = \
            [read_track(sim, reader, subhalos, cat, i, **read_kwargs)[:-1]
             for i in indices]
    else:
        track_data = [[] for _ in indices]
        pool = mp.Pool(args.ncores)
        # need to pass the indices this way so I can keep track of order
        results = \
            [pool.apply_async(
                read_track, args=(sim,reader,subhalos,cat[indices],i),
                kwds=read_kwargs)
             for i in range(len(indices))]
        pool.close()
        pool.join()
        for out in results:
            out = out.get()
            track_data[out[-1]] = out[:-1]
    _ = [plot_track(axes, sim, data, color=color, label_host=label_host,
                    show_history=show_history, label_history=label_history)
         for data, color in zip(track_data, colors)]
    return


def read_tracks(args, sim, reader, cat, indices, nsub=10, sort_mass=True,
                **kwargs):
    """
    if `sort_mass==True`, then tracks are sorted per host by final
    total mass. Note that this assumes that tracks belonging to
    each host are all together one after the other (will generalize
    later)
    """
    if not hasattr(indices, '__iter__'):
        indices = [indices]
    to = time()
    if args.ncores == 1:
        tracks = \
            [read_track(sim, reader, cat, i, **kwargs)[:-1]
             for i in indices]
    else:
        tracks = [[] for _ in indices]
        pool = mp.Pool(args.ncores)
        # need to pass the indices this way so I can keep track of order
        results = \
            [pool.apply_async(
                 read_track, args=(sim,reader,cat[indices],i),
                 kwds=kwargs)
             for i in range(len(indices))]
        pool.close()
        pool.join()
        for out in results:
            out = out.get()
            tracks[out[-1]] = out[:-1]
    print('Read all tracks in {0:.2f} minutes'.format((time()-to)/60))
    #read_track output (last value not recorded):
    #return trackid, t, Mt, rank, depth, icent, isat, iinf, \
        #hostid, th, Mh, track_idx
    to = time()
    if sort_mass:
        hostids = np.array([t[-3] for t in tracks])
        nh = np.arange(hostids.size, dtype=int)
        # this is the last value of the last Mt column
        Mtot_now = np.array([t[2][-1][-1] for t in tracks])
        for i in np.unique(hostids):
            mask = (hostids == i)
            aux = [tracks[m] for m in nh[mask]]
            jsort = np.argsort(-Mtot_now[mask])
            for j, k in enumerate(nh[mask]):
                tracks[k] = aux[jsort[j]]
    print('Sorted tracks in {0:.2f} s'.format(time()-to))
    return tracks


def read_track(sim, reader, cat, track_idx, load_history=True,
               massindex=None, verbose=True):
    trackid = cat['TrackId'][track_idx]
    ti = time()
    track = Track(reader.GetTrack(trackid), sim)
    if verbose:
        print('Loaded track #{2} (TrackID {0}) in {1:.2f} minutes'.format(
            trackid, (time()-ti)/60, track_idx))
    # this is what takes up all the time, especially the infall
    if load_history:
        # las moment it was a central
        icent = track.last_central_snapshot_index
        # first time it was a satellite
        isat = track.first_satellite_snapshot_index
        # infall
        iinf = track.infall('index')
    else:
        icent = isat = iinf = None
    host = track.host(-1, return_value='track')
    host = Track(host, sim)
    t = track.lookback_time()
    th = host.lookback_time()
    if massindex is None:
        Mt = [track.mass(index=i) for i in range(6)]
        Mt.append(track.mass(index=-1))
        Mt = np.array(Mt)
        Mh = [host.mass(index=i) for i in range(6)]
        Mh.append(host.mass(index=-1))
        Mh = np.array(Mh)
    else:
        Mt = track.mass(index=massindex)
        Mh = host.mass(index=massindex)
    #print('Mt =', Mt.shape)
    #print(
    rank = track.track['Rank']
    depth = track.track['Depth']
    hostid = host.trackid
    return trackid, t, Mt, rank, depth, icent, isat, iinf, \
        hostid, th, Mh, track_idx


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


def setup_track_axes(axes, sim, lookback_cosmo=False, textcolor='0.3',
                     masslabel=r'M_\mathrm{total}', is_last_row=True):
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
    ylabel = [r'${1}({0})$'.format(x, masslabel),
              r'${1}({0})/{1}$$_{{,0}}$'.format(x, masslabel)]
    axes[0].set_ylabel(ylabel[0])
    axes[1].set_ylabel(ylabel[1])
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




