from icecream import ic
from matplotlib import pyplot as plt, ticker
import multiprocessing as mp
import numpy as np
import os
from scipy.stats import (
    binned_statistic as binstat, binned_statistic_dd as binstat_dd)
import sys
from time import time

from plottery.plotutils import colorscale, savefig, update_rcParams
update_rcParams()

from ..hbt_tools import save_plot
#from hbtpy.hbtplots import RelationPlotter
#from hbtpy.hbtplots.core import ColumnLabel
from .plot_auxiliaries import (
    binning, definitions, get_axlabel, get_bins, get_label_bincol, logcenters,
    massbins, plot_line)
from .plot_definitions import (
    ccolor, scolor, massnames, units, xbins, binlabel, events, axlabel)


def run(sim, subs, logM200Mean_min, which_relations, ncores=1):
    def print_header(relation_type):
        print(f'\n########\n### {relation_type} relations\n########')

    use_mp = (ncores > 1)
    # just using for labels for now
    #plotter = RelationPlotter(sim, subs)
    #label = ColumnLabel()

    pool = mp.Pool(ncores) if use_mp else None
    # x-axis: time since historical event
    if which_relations['time']:
        print_header('time')
        wrap_relations_time(sim, subs, pool)

    #return

    # x-axis: cluster-centric distance
    if which_relations['distance']:
        print_header('distance')
        wrap_relations_distance(sim, subs, pool)

    relation_kwargs = dict(
        satellites_label='Present-day satellites',
        centrals_label='Present-day centrals',
        min_hostmass=logM200Mean_min, xlim=(3e7,1e12))

    if which_relations['hsmr history']:
        print_header('HSMR history')
        wrap_relations_hsmr_history(sim, subs, pool, **relation_kwargs)
    # historical HSMR binned by present-day quantities

    # present-day HSMR binned by historical quantities
    relation_kwargs['satellites_label'] = 'All satellites'
    relation_kwargs['centrals_label'] = 'Centrals'
    if which_relations['hsmr']:
        print_header('HSMR')
        wrap_relations_hsmr(sim, subs, pool, **relation_kwargs)

    if use_mp:
        pool.close()
        pool.join()



def do_wrap_relations(pool, *args, **kwargs):
    if pool is None:
        wrap_relations(*args, **kwargs)
    else:
        pool.apply_async(wrap_relations, args=args, kwds=kwargs)
    return


def wrap_relations(sim, subs, xcol, ycol, bincol, x_bins=10, bins=None,
                   xscale='log', logbins=None,
                   show_satellites=False, **kwargs):
    label = get_label_bincol(bincol)
    if logbins is None:
        logbins = \
            not ('time' in bincol \
                 or ('/' in bincol \
                     and (bincol.count('Mbound') == 2 \
                          or bincol.count('Mstar') == 2)))
    if bins is None:
        bins = get_bins(bincol, logbins)
    for statistic in ('std', 'mean', 'std/mean'):
        #showsat = (statistic == 'mean') * show_satellites
        # since this is the last one we run there shouldn't be
        # any problem in overwriting these
        if statistic == 'std/mean':
            kwargs['yscale'] = 'linear'
            kwargs['ylim'] = (0, 1.5)
        plot_relation(
            sim, subs, xcol=xcol, ycol=ycol, xbins=x_bins,
            xscale=xscale, bincol=bincol,
            statistic=statistic, binlabel=label, bins=bins,
            logbins=logbins, xlabel=get_axlabel(xcol, 'mean'),
            ylabel=get_axlabel(ycol, statistic),
            show_satellites=show_satellites, **kwargs)
            #selection=selection, selection_min=selection_min,
            #show_satellites=show_satellites, show_centrals=show_centrals)
    return


def wrap_relations_distance(sim, subs, pool):
    xscale = 'log'
    xb = np.logspace(-2, 0.5, 9) if xscale == 'log' \
        else np.linspace(0, 3, 9)
    d = 'ComovingMostBoundDistance'
    for event in ('last_infall', 'first_infall', 'cent', 'sat'):
        h = f'history:{event}'
        for ycol in ('Mbound/Mstar', 'Mstar', 'Mbound', 'Mstar/Mbound',
                     f'Mstar/{h}:Mbound', f'Mbound/{h}:Mbound',
                     f'Mstar/{h}:Mstar', f'Mbound/{h}:Mstar'):
            if ycol == 'Mbound/Mstar':
                ylim = (2.5, 110)
            else: ylim = None
            for xyz in ('', 0, 1, 2):
                for xcol in (f'{d}{xyz}', f'{d}{xyz}/R200Mean'):
                    for bincol in ('Mbound', 'Mstar', f'{h}:Mstar',
                                   f'{h}:Mbound', f'{h}:time'):
                        logbins = ('time' not in bincol)
                        kwds = dict(
                            x_bins=xb, xscale=xscale,
                            bins=get_bins(bincol, logbins),
                            selection='Mstar', selection_min=1e8, ylim=ylim,
                            show_satellites=True, show_centrals=False)
                        do_wrap_relations(
                            pool, sim, subs, xcol, ycol, bincol, **kwds)
                        #break
    return


def wrap_relations_hsmr(sim, subs, pool, **relation_kwargs):
    x_bins = np.logspace(7.5, 11.7, 9)
    events = [f'history:{e}'
              for e in ('first_infall', 'last_infall', 'cent', 'sat')]
    history_bincols = [
        (f'{h}:Mbound', f'{h}:Mstar', f'{h}:Mdm', f'{h}:Mstar/{h}:Mbound',
         f'{h}:Mbound/{h}:Mstar', f'{h}:Mdm/{h}:Mbound', f'{h}:Mstar/{h}:Mdm',
         f'Mbound/{h}:Mbound', f'Mstar/{h}:Mbound', f'Mdm/{h}:Mdm',
         f'{h}:time')
        for h in events]
    bincols = ['ComovingMostBoundDistance','ComovingMostBoundDistance0',
               'LastMaxMass', 'Mbound/LastMaxMass', 'Mstar/LastMaxMass',
               'M200Mean']#, 'Mbound/M200Mean']
    bincols = bincols + [col for cols in history_bincols for col in cols]
    bincols = ['Mstar/history:last_infall:Mbound']
    ic(bincols)
    #for ycol in ('Mbound', 'Mbound/Mstar', 'Mtotal', 'Mtotal/Mstar'):
    for ycol in ('Mbound/Mstar',):
        for bincol in bincols:
            if bincol in xbins:
                bins = xbins[bincol]
            else: bins = 6
            kwds = {**dict(bins=bins, show_satellites=False,
                           show_centrals=True,
                           #show_centrals=('history' not in ycol)),
                           #ylim=(5e8,2e14)),
                           ),
                    **relation_kwargs}
            do_wrap_relations(
                pool, sim, subs, 'Mstar', ycol, bincol, x_bins, **kwds)
    return


def wrap_relations_hsmr_history(sim, subs, pool, xlim=None, **relation_kwargs):
    for event in ('last_infall', 'first_infall', 'cent', 'sat'):
        h = f'history:{event}'
        bincols = ['Mbound', 'Mstar', 'Mstar/Mbound', 'Mbound/Mstar',
                   'ComovingMostBoundDistance','ComovingMostBoundDistance0',
                   'LastMaxMass', 'Mbound/LastMaxMass', 'Mstar/LastMaxMass',
                   'M200Mean', 'Mbound/M200Mean', f'{h}:time']
        for bincol in bincols:
            logbins = ('time' not in bincol)
            bins = get_bins(bincol, logbins)
            kwds = {**dict(bins=bins), **relation_kwargs}
            for ycol in (f'{h}:Mbound', f'{h}:Mbound/{h}:Mstar',
                         f'Mbound/{h}:Mbound'):
                x_bins = np.logspace(7.5, 11.7, 9)
                do_wrap_relations(
                    pool, sim, subs, 'Mstar', ycol, bincol, x_bins, bins=bins,
                    xlim=xlim)
                do_wrap_relations(
                    pool, sim, subs, f'{h}:Mstar', ycol, bincol, x_bins,
                    xlim=xlim, **kwds)
                x_bins = np.logspace(-4, -0.7, 11)
                do_wrap_relations(
                    pool, sim, subs, f'Mstar/{h}:Mbound', ycol, bincol, x_bins,
                    bins=bins, xlim=xlim)
        for bincol in (f'{h}:time', 'M200Mean', 'Mbound/M200Mean', 'Mstar',
                       f'{h}:Mstar'):
            logbins = ('time' not in bincol)
            bins = get_bins(bincol, logbins)
            x_bins = np.logspace(0.7, 3, 10)
            do_wrap_relations(
                pool, sim, subs, f'{h}:Mbound/{h}:Mstar', 'Mbound/Mstar',
                bincol, x_bins, logbins=logbins, xlim=None, **relation_kwargs)
            x_bins = np.logspace(9, 13, 10)
            for ycol in ('Mbound', 'Mstar'):
                do_wrap_relations(
                    pool, sim, subs, f'{h}:Mbound', ycol, bincol, x_bins,
                    xlim=(x_bins[0],x_bins[-1]), **relation_kwargs)
    return


def wrap_relations_time(sim, subs, pool):
    #x_bins = np.arange(0, 14, 1)
    for event in ('last_infall', 'first_infall', 'cent', 'sat'):
        h = f'history:{event}'
        xcol = f'{h}:time'
        for ycol in ('Mbound/Mstar', f'Mbound/{h}:Mbound',
                     f'Mstar/{h}:Mbound'):
            for bincol in (f'{h}:Mbound/{h}:Mstar', f'{h}:Mstar/{h}:Mbound',
                           'M200Mean', 'Mbound/Mstar', 'Mstar/Mbound',
                           'Mstar', 'Mbound', f'{h}:Mbound', f'{h}:Mstar'):
                if 'Mstar' in bincol:
                    bins = np.logspace(8, 12, 7)
                else:
                    bins = get_bins(bincol)
                kwds = dict(
                    bins=bins, x_bins=xbins['time'], logbins=True,
                    selection='Mstar', selection_min=1e8, xscale='linear',
                    show_satellites=True)
                do_wrap_relations(pool, sim, subs, xcol, ycol, bincol, **kwds)
    return


################################################
################################################


def plot_relation(sim, subs, xcol='Mstar', ycol='Mbound',
                  statistic='mean', selection=None,
                  selection_min=None, selection_max=None, xlim=None,
                  ylim=None, xbins=10, xscale='log', yscale='log',
                  hostmass='M200Mean', min_hostmass=13, show_hist=True,
                  bindata=None, bincol=None, bins=6, logbins=False,
                  binlabel='', mask=None, xlabel=None, ylabel=None,
                  show_satellites=True, show_centrals=False,
                  satellites_label='All satellites',
                  centrals_label='Centrals', literature=True):
    """Plot the SHMR and HSMR

    ``bincol`` and ``bins`` allow the relations to be binned in a
    third quantity

    """
    cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat = \
        definitions(subs, hostmass=hostmass, min_hostmass=min_hostmass)
    xdata = subs[xcol]
    ydata = subs[ycol]
    if bincol is not None:
        bindata = subs[bincol]
        if not np.iterable(bins):
            j = np.isfinite(bindata)
            if logbins:
                j = j & (bindata > 0)
                vmin, vmax = np.log10(np.percentile(bindata[j], [1,99]))
                bins = np.logspace(vmin, vmax, bins)
            else:
                vmin, vmax = np.percentile(bindata[j], [1,99])
                bins = np.linspace(vmin, vmax, bins)
        ic(bincol, logbins)
        ic(bins)
        ic(bindata.min(), bindata.max())
        ic(np.histogram(bindata, bins)[0])
        if not binlabel:
            binlabel = f'{statistic}({bincol})'
        if logbins:
            lb = np.log10(bins)
            bin_centers = 10**((lb[1:]+lb[:-1])/2)
        else:
            bin_centers = (bins[1:]+bins[:-1]) / 2
        colors, cmap = colorscale(array=bin_centers, log=logbins)
    mask = np.isfinite(xdata) & np.isfinite(ydata) & np.isfinite(bindata)
    if selection is not None:
        seldata = subs[selection]
        if selection_min is not None:
            mask = mask & (seldata >= selection_min)
        if selection_max is not None:
            mask = mask & (seldata <= selection_max)
    ic(mask.sum())
    xdata = xdata[mask]
    ydata = ydata[mask]
    bindata = bindata[mask]
    sat = sat[mask]
    dark = dark[mask]
    cen = cen[mask]
    gsat = sat & ~dark
    ic(xbins)
    if isinstance(xbins, int):
        if xscale == 'log':
            gsat = gsat & (xdata > 0)
            xbins = np.linspace(
                np.log10(xdata[gsat].min()), np.log10(xdata[gsat].max()),
                xbins+1)
        else:
            xbins = np.linspace(xdata[gsat].min(), xdata[gsat].max(), xbins+1)
        if xscale == 'log':
            xbins = 10**xbins
    if xscale == 'log':
        xb = np.log10(xbins)
        xcenters = 10**((xb[:-1]+xb[1:])/2)
    else:
        xcenters = (xbins[:-1]+xbins[1:]) / 2
    ic(xcol, xbins)
    ic(xdata.min(), xdata.max())
    ic(np.histogram(xdata, xbins)[0])
    ic(xcenters)
    ic(xlim, ylim)
    logx = np.log10(xcenters)

    lw = 4
    # as a function of third variable
    fig, ax = plt.subplots(figsize=(8,6))
    # fix bins here
    if '/' in statistic:
        stat = statistic.split('/')
        relation = binstat_dd(
                [bindata[gsat], xdata[gsat]], ydata[gsat], stat[0],
                [bins,xbins]).statistic \
            / binstat_dd(
                [bindata[gsat], xdata[gsat]], ydata[gsat], stat[1],
                [bins,xbins]).statistic
    else:
        relation = binstat_dd(
            [bindata[gsat], xdata[gsat]], ydata[gsat], statistic,
            [bins,xbins]).statistic
    # this will make the plot look nicer
    if 'std' in statistic:
        relation[relation == 0] = np.nan
    ic(relation)
    ic(relation.shape)
    for i in range(bins.size-1):
        ax.plot(xcenters, relation[i], '-', color=colors[i],
                lw=4, zorder=10+i)
    cbar = plt.colorbar(cmap, ax=ax)
    cbar.set_label(binlabel)
    if logbins:
        cbar.ax.set_yscale('log')
        if bins[0] >= 0.001 and bins[-1] <= 1000:
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%s'))
    # compare to overall and last touches
    if show_satellites:
        j = mask & gsat
        if '/' in statistic:
            st = statistic.split('/')
            satrel = binstat(xdata[j], ydata[j], st[0], xbins)[0] \
                / binstat(xdata[j], ydata[j], st[1], xbins)[0]
        else:
            satrel = binstat(xdata[j], ydata[j], statistic, xbins)[0]
        plot_line(
            ax, xcenters, satrel, marker='+', lw=3, color=scolor,
            label=satellites_label, zorder=100)
    if show_centrals:
        j = mask & cen
        if j.sum() == 0:
            show_centrals = False
        else:
            if '/' in statistic:
                st = statistic.split('/')
                cenrel = binstat(xdata[j], ydata[j], st[0], xbins)[0] \
                    / binstat(xdata[j], ydata[j], st[1], xbins)[0]
            else:
                cenrel = binstat(xdata[j], ydata[j], statistic, xbins)[0]
            plot_line(
                ax, xcenters, cenrel, marker='x', lw=3, color=ccolor,
                label=centrals_label, zorder=100)
    if literature:
        xcol_split = xcol.split(':')
        if len(xcol_split) <= 3 and xcol_split[-1] == 'Mstar':
            xlit = 10**np.array([9.51, 10.01, 10.36, 10.67, 11.01])
            #ylit = read_literature('sifon18_mstar', 'Msat_rbg')
        elif 'Distance' in xcol:
            # Rsat (Mpc) - missing normalization
            xlit = np.array([0.23, 0.52, 0.90, 1.55])
            #ylit = read_literature('sifon18_Rbcg', 'Msat_rbg')
    if show_satellites or show_centrals:
        ax.legend(fontsize=18)
    if xlabel is None:
        #xlabel = r'$\log\,{0}$'.format(sim.masslabel(mtype='stars'))
        xlabel = xcol
    if ylabel is None:
        #ylabel = r'$\log\,{0}$'.format(sim.masslabel(mtype='total'))
        ylabel = ycol
    # cheap hack
    xlabel = xlabel.replace('$$', '$')
    ylabel = ylabel.replace('$$', '$')
    ax.set(xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale,
           xlim=xlim, ylim=ylim)
    if 'Distance' in xcol and xscale == 'log':
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%s'))
    if 'time' in xcol:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    # add a random number to see whether they update
    #ax.annotate(
        #f'{np.random.randint(1000)}', xy=(0.96,0.96), xycoords='axes fraction',
        #fontsize=14, va='top', ha='right', color='C3')
    # format filename
    bincol = bincol.replace('/', '-over-').replace(':', '-')
    xcol = xcol.replace('/', '-over-').replace(':', '-')
    ycol = ycol.replace('/', '-over-').replace(':', '-')
    statistic = statistic.replace('/', '-over-')
    outcols = f'{xcol}_{ycol}'
    outname = f'{statistic}__{outcols}'
    output = os.path.join(
        'relations', ycol, outcols, outname,
        f'{outname}__bin__{bincol}')
    save_plot(fig, output, sim)
    return relation
