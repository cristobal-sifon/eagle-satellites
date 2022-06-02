import cmasher as cmr
from icecream import ic
from matplotlib import cm, colors as mplcolors, pyplot as plt, ticker
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import multiprocessing as mp
import numpy as np
import os
from scipy.stats import (
    binned_statistic as binstat, binned_statistic_dd as binstat_dd)
import sys
from time import time
import warnings

from plottery.plotutils import colorscale, savefig, update_rcParams
from plottery.statsplots import contour_levels
update_rcParams()

from ..hbt_tools import save_plot
#from hbtpy.hbtplots import RelationPlotter
#from hbtpy.hbtplots.core import ColumnLabel
from .plot_auxiliaries import (
    binning, definitions, get_axlabel, get_bins, get_label_bincol, logcenters,
    massbins, plot_line)
from .plot_definitions import (
    ccolor, scolor, massnames, units, xbins, binlabel, events, axlabel, ylims)

warnings.simplefilter('ignore', RuntimeWarning)


def run(args, sim, subs, logM200Mean_min, which_relations, ncores=1):
    def print_header(relation_type):
        print(f'\n########\n### {relation_type} relations\n########')

    use_mp = (ncores > 1)
    # just using for labels for now
    #plotter = RelationPlotter(sim, subs)
    #label = ColumnLabel()

    plot_args = []
    pool = mp.Pool(ncores) if use_mp else None
    # x-axis: time since historical event
    for stat in args.stats:
        if 'time' in which_relations:
            print_header('time')
            out = wrap_relations_time(args, stat)
            plot_args.extend(out)

        # x-axis: cluster-centric distance
        if 'distance' in which_relations:
            print_header('distance')
            out = wrap_relations_distance(args, stat)
            plot_args.extend(out)

        relation_kwargs = dict(
            satellites_label='Satellites today',
            centrals_label='Centrals today', show_centrals=True,
            min_hostmass=logM200Mean_min)#, xlim=(3e7,1e12))

        # historical HSMR binned by present-day quantities
        if 'hsmr-history' in which_relations:
            print_header('HSMR history')
            out = wrap_relations_hsmr_history(
                args, stat, **relation_kwargs)
            plot_args.extend(out)

        # present-day HSMR binned by historical quantities
        relation_kwargs['satellites_label'] = 'All satellites'
        relation_kwargs['centrals_label'] = 'Centrals'
        if 'hsmr' in which_relations:
            print_header('HSMR')
            out = wrap_relations_hsmr(args, stat, **relation_kwargs)
            plot_args.extend(out)

    print()
    print()
    print(f"Producing {len(plot_args)} plots, let's go!")
    print()
    print()

    # ic(len(args))
    # for args_i, kwargs_i in args:
    # #for args_i, kwargs_i in zip(args[:3], kwargs[:3]):
    #     ic(args_i[0], kwargs_i['idx'], kwargs_i.get('logbins'))
    # return

    if use_mp:
        _ = [pool.apply_async(wrap_relations, args=(args,sim,subs,*args_i),
                              kwds=kwargs_i)
             for args_i, kwargs_i in plot_args]
        pool.close()
        pool.join()
    else:
        _ = [wrap_relations(args, sim, subs, *args_i, **kwargs_i)
             #for args_i, kwargs_i in zip(args[:1], kwargs[:1])]
             for args_i, kwargs_i in plot_args]

    return


def wrap_relations(args, sim, subs, xcol, ycol, bincol, x_bins=10, bins=None,
                   xscale='log', logbins=None, show_satellites=False,
                   statistic='mean', **kwargs):
    label = get_label_bincol(bincol)
    #print('\n'*5)
    ic()
    ic(bincol, bins, logbins)
    if logbins is None:
        if bincol is not None:
            logbins = \
                not ('time' in bincol \
                     or ('/' in bincol \
                         and (bincol.count('Mbound') == 2 \
                              or bincol.count('Mstar') == 2)))
        else:
            logbins = False
    if bins is None:
        bins = get_bins(bincol, logbins)
    #showsat = (statistic == 'mean') * show_satellites
    # since this is the last one we run there shouldn't be
    # any problem in overwriting these
    #if statistic == 'count' or bincol is None:
        #kwargs['yscale'] = 'linear'
        #kwargs['ylim'] = None
    if 'yscale' not in kwargs:
        kwargs['yscale'] = 'log' if statistic == 'mean' else 'linear'
    if statistic != 'mean':
        kwargs['ylim'] = (0, 0.8) if statistic == 'std' else (0, 1.5)
    plot_relation(
        sim, subs, xcol=xcol, ycol=ycol, xbins=x_bins,
        xscale=xscale, bincol=bincol,
        statistic=statistic, binlabel=label, bins=bins,
        logbins=logbins, xlabel=get_axlabel(xcol, 'mean'),
        ylabel=get_axlabel(ycol, statistic), cmap='viridis',
        show_satellites=show_satellites, **kwargs)
        #selection=selection, selection_min=selection_min,
        #show_satellites=show_satellites, show_centrals=show_centrals)
    return


def wrap_relations_distance(args, stat):
    xscale = 'log'
    xb = np.logspace(-2, 0.5, 9) if xscale == 'log' \
        else np.linspace(0, 3, 9)
    d = 'ComovingMostBoundDistance'
    plot_args = []
    # defined here to avoid repetition
    ycols_present = ['Mbound/Mstar', 'Mstar/Mbound', 'Mstar', 'Mbound',
                     'Mbound/M200Mean']
    bincols_present = ['M200Mean', 'Mbound', 'Mstar']
    #ylim = lambda yc: (2.5, 110) if yc == 'Mbound/Mstar' else None
    for ie, event in enumerate(('first_infall', 'last_infall', 'cent', 'sat')):
        h = f'history:{event}'
        ycols = [f'Mbound/{h}:Mbound', f'Mstar/{h}:Mstar',
                 f'{h}:time', f'{h}:Mbound/{h}:Mstar',
                 f'{h}:Mbound', f'{h}:Mstar'] \
                + ycols_present
        bincols = [f'{h}:time', f'{h}:Mstar', f'{h}:Mbound',
                   f'{h}:Mbound/{h}:Mstar'] \
                  + bincols_present
        # uncomment when testing
        #ycols = ['Mbound/Mstar']
        #bincols = ['Mstar']
        for ycol in ycols:
            ylim = (2, 150) \
                if ycol in ('Mbound/Mstar', f'{h}:Mbound/{h}:Mstar') else None
            #ylim = None
            for xyz in ('', 0):
                for xcol in (f'{d}{xyz}', f'{d}{xyz}/R200MeanComoving'):
                #for xcol in (f'{d}{xyz}/R200MeanComoving',):
                    # 2d histogram
                    if stat == 'count':
                        kwds = dict(
                            x_bins=np.logspace(-2, 0.7, 26), ybins=20,
                            selection='Mstar', selection_min=1e8,
                            xscale='log', statistic='count', bins=None,
                            lines_only=False, ylim=ylims.get(ycol))
                        plot_args.append([[xcol, ycol, None], kwds.copy()])
                    # line plots
                    else:
                        kwds = dict(
                            x_bins=xb, xscale=xscale,
                            selection='Mstar', selection_min=1e8, ylim=ylim,
                            show_satellites=False, show_centrals=False,
                            show_ratios=False, statistic=stat)
                        for bincol in bincols:
                            # avoid repetition and trivial plots
                            if (ie > 0 and bincol in bincols_present \
                                    and ycol in ycols_present) \
                                    or bincol == ycol:
                                continue
                            logbins = ('time' not in bincol)
                            kwds['bins'] = get_bins(bincol, logbins, n=5)
                            if ycol == 'Mbound/Mstar' and bincol == 'M200Mean':
                                kwds['ylim_ratios'] = (0.5, 1.5)
                            else:
                                kwds['ylim_ratios'] = None
                            plot_args.append(
                                [[xcol, ycol, bincol], kwds.copy()])
                            #wrap_relations(sim, subs, *args[-1], **kwargs[-1])
        #break
    return plot_args


def wrap_relations_hsmr(args, stat, **relation_kwargs):
    x_bins = {'Mstar': np.logspace(7.5, 11.7, 10),
              'Mbound': np.logspace(8.5, 12.7, 10)}
    events = [f'history:{e}'
              for e in ('first_infall', 'last_infall', 'cent', 'sat')]
    history_bincols = [
        (f'{h}:Mbound', f'{h}:Mstar', f'{h}:Mdm', f'{h}:Mstar/{h}:Mbound',
         f'{h}:Mbound/{h}:Mstar', f'{h}:Mdm/{h}:Mbound', f'{h}:Mstar/{h}:Mdm',
         f'Mbound/{h}:Mbound', f'Mstar/{h}:Mbound', f'Mdm/{h}:Mdm',
         f'{h}:time')
        for h in events]
    bincols = ['ComovingMostBoundDistance', 'ComovingMostBoundDistance0',
               'ComovingMostBoundDistance/R200MeanComoving',
               'ComovingMostBoundDistance0/R200MeanComoving',
               'LastMaxMass', 'Mbound/LastMaxMass', 'Mstar/LastMaxMass',
               'M200Mean', 'Mbound/M200Mean', 'Mbound']
    bincols = bincols + [col for cols in history_bincols for col in cols]
    bincols = [i for i in bincols if 'time' in i]
    #bincols = ['Mstar/history:last_infall:Mbound']
    #bincols = ['ComovingMostBoundDistance/R200MeanComoving',
               #'ComovingMostBoundDistance0/R200MeanComoving']
    #bincols = ['Mbound']
    ic(bincols)
    #for ycol in ('Mbound', 'Mbound/Mstar', 'Mtotal', 'Mtotal/Mstar'):
    plot_args = []
    for ycol, ylim in zip(('Mstar/Mbound', 'Mbound/Mstar', 'Mbound'),
                          ((2e-3,0.5), (3,300), (5e8,2e14))):
        # histograms
        if stat == 'count':
            kwds = dict(x_bins=20, ybins=20,
                        #selection='Mstar', selection_min=1e8,
                        statistic='count', lines_only=False,
                        show_centrals=True)
            args.append([['Mstar', ycol, None], kwds.copy()])
        # lines
        else:
            for bincol in bincols:
                #if bincol != 'M200Mean': continue
                logbins = ('time' not in bincol)
                if ':time' in bincol:
                    bins = np.arange(0, 11.1, 2.2)
                else:
                    bins = get_bins(bincol, logbins, n=5)
                kwds = {**dict(bins=bins, ylim=ylim, statistic=stat,
                               show_ratios=False,
                               show_satellites=False, show_centrals=True),
                        **relation_kwargs}
                for xcol in ('Mstar', 'Mbound'):
                    if ycol == xcol or xcol == bincol:
                        continue
                    xb = xbins.get(xcol)
                    plot_args.append([[xcol, ycol, bincol, xb], kwds.copy()])
                #break
        #break
    return plot_args


def wrap_relations_hsmr_history(args, stat, do_mass=True, do_ratios=False,
                                xlim=None, **relation_kwargs):
    """Plot historical quantities on the x-axis"""
    plot_args = []
    kwds = dict(statistic=stat, **relation_kwargs)
    ii = 0
    if not np.any(np.in1d([stat], ('mean','std','std/mean'))):
        return []
    for event in ('first_infall', 'last_infall', 'cent', 'sat'):
    #for event in ('cent',):
        h = f'history:{event}'
        # ratios in the x-axis
        if do_ratios:
            kwds['show_centrals'] = False
            for bincol in (f'{h}:z', f'{h}:time', 'M200Mean', 'Mbound/M200Mean',
                           'Mstar', f'{h}:Mstar'):
            #for bincol in (f'{h}:time',):
                logbins = ('time' not in bincol)
                ic(bincol, logbins)
                if ':time' in bincol:
                    bins = np.arange(0, 11.1, 2.2)
                else:
                    bins = get_bins(bincol, logbins, n=5)
                xcol = (f'{h}:Mbound/Mbound', f'{h}:Mstar/Mstar',
                        f'{h}:Mstar/Mstar',  f'{h}:Mbound/{h}:Mstar')
                ycol = (f'{h}:Mstar/Mstar', f'{h}:Mbound/Mbound',
                        f'{h}:Mdm/Mdm', 'Mbound/Mstar')
                x_bins = [np.logspace(-0.5, 1, 11), np.logspace(-0.5, 1, 11),
                          np.logspace(-0.5, 1, 11), np.logspace(0.7, 3, 10)]
                kwds['logbins'] = logbins
                kwds['show_1to1'] = True
                for xc, yc, xb in zip(xcol, ycol, x_bins):
                    kwds['xlim'] = (xb[0], xb[-1])
                    kwds['ylim'] = (xb[0], xb[-1])
                    args.append([[xc, yc, bincol, xb], kwds.copy()])
                #
                x_bins = np.logspace(9, 13, 10)
                kwds['xlim'] = (x_bins[0], x_bins[-1])
                kwds['show_1to1'] = False
                for ycol in ('Mbound', 'Mstar'):
                    plot_args.append(
                        [[f'{h}:Mbound', ycol, bincol, x_bins], kwds.copy()])
        # stellar mass in the x-axis
        if do_mass:
            bincols = (f'{h}:z', f'{h}:time',
                       'Mbound', 'Mstar', 'Mstar/Mbound', 'Mbound/Mstar', 'Mdm/Mstar',
                       'ComovingMostBoundDistance','ComovingMostBoundDistance0',
                       'LastMaxMass', 'Mbound/LastMaxMass', 'Mstar/LastMaxMass',
                       'M200Mean', 'Mbound/M200Mean')
            ycols = (f'{h}:Mbound', f'{h}:Mbound/{h}:Mstar',
                        f'Mbound/{h}:Mbound')
            ylims = ((1e9, 1e13), (3, 300), (0.01, 2))
            bincols = bincols[:2]
            for bincol in bincols:
                logbins = ('time' not in bincol \
                    and bincol.split(':')[-1] != 'z')
                ic(bincol, logbins)
                if ':time' in bincol:
                    bins = np.arange(0, 11.1, 2.2)
                else:
                    bins = get_bins(bincol, logbins,
                                    n=None if bincol.split(':')[-1] == 'z' else 5)
                kwds = {**dict(bins=bins, statistic=stat, logbins=logbins),
                        **relation_kwargs}
                for ycol, ylim in zip(ycols, ylims):
                    for xcol in ('Mstar', f'{h}:Mstar'):
                        xb = xbins[xcol.split(':')[-1]]
                        kwds['ylim'] = ylim
                        plot_args.append(
                            [[xcol, ycol, bincol, xb], kwds.copy()])
                #break
    return plot_args


def wrap_relations_time(args, stat):
    #x_bins = np.arange(0, 14, 1)
    plot_args = []
    kwds_count = dict(
        x_bins=30, ybins=30, selection='Mstar', selection_min=1e8,
        xscale='linear', statistics='count', yscale='log', lines_only=False)
    kwds = dict(
        x_bins=xbins['time'], logbins=True, selection='Mstar',
        statistic=stat, xscale='linear', yscale='log', selection_min=1e8,
        show_satellites=False, show_ratios=False)
    events = ('cent', 'sat', 'first_infall', 'last_infall')
    for i_e, event in enumerate(events):
        h = f'history:{event}'
        time_diffs = [f'{h}:time-history:{e}:time' for e in events[:i_e]]
        xcols = [f'{h}:time'] + time_diffs # redshift is giving trouble, leave for later
        for xcol in xcols:
        #xcol = f'{h}:time'
            # histograms
            if stat == 'count':
                # for ycol in ('Mbound', 'Mbound/Mstar', f'{h}:Mbound',
                #              f'Mbound/{h}:Mbound', f'Mstar/{h}:Mbound',
                #              f'Mstar/{h}:Mstar', 'M200Mean', 'Mbound/M200Mean',
                #              f'Mdm/{h}:Mdm', f'Mgas/{h}:Mgas',
                #              f'Mstar/Mbound', f'Mdm/Mbound', f'Mgas/Mbound',
                #              'ComovingMostBoundDistance/R200MeanComoving'):
                for ycol in (f'{h}:time',
                             f'history:first_infall:Mdm/{h}:Mdm',
                             f'history:first_infall:Mstar/{h}:Mstar'):
                    kwds_count['ylim'] = ylims.get(ycol)
                    plot_args.append([[xcol, ycol, None], kwds_count.copy()])
            # lines
            else:
                ycols = [f'Mdm/{h}:Mdm', f'Mstar/{h}:Mstar', 'Mbound/Mstar',
                         f'Mbound/{h}:Mbound', f'{h}:Mbound/{h}:Mstar',
                         f'Mstar/{h}:Mbound', 'Mbound/M200Mean',
                         'ComovingMostBoundDistance/R200MeanComoving']
                if xcol.count('time') == 1:
                    ycols += time_diffs
                ycols += [f'{h}:{m}/history:{e}:{m}'
                          for e in events[:i_e] for m in ('Mbound','Mdm','Mstar')] 
                bincols = [f'{h}:time',
                           f'{h}:Mbound/{h}:Mstar', f'{h}:Mstar/{h}:Mbound',
                           'M200Mean', 'Mbound/Mstar', 'Mstar/Mbound',
                           'Mstar', 'Mbound', f'{h}:Mbound', f'{h}:Mstar']
                for ycol in ycols[:3]:
                    for bincol in bincols[:3]:
                        if bincol == ycol or bincol == xcol:
                            continue
                        bins = np.logspace(8, 12, 7) \
                            if ('Mstar' in bincol and '/' not in bincol) \
                            else get_bins(bincol)
                        kwds['yscale'] = 'linear' \
                            if ycol in (f'Mstar/{h}:Mstar', f'Mdm/{h}:Mdm') \
                            else 'log'
                        kwds['yscale'] = 'linear'
                        kwds['ylim'] = (0, 1) if ycol == f'Mdm/{h}:Mdm' \
                            else None
                        kwds['bins'] = bins
                        kwds['logbins'] = ('time' not in bincol)
                        kwds['show_centrals'] = (not h in ycol)
                        plot_args.append([[xcol, ycol, bincol], kwds.copy()])
        #break
    return plot_args


################################################
################################################

def do_xbins(X, mask, xbins, xlim=None, xscale='log'):
    ic()
    ic(xlim)
    mask = mask & np.isfinite(X)
    if xlim is not None:
        X = X[(xlim[0] <= X) & (X <= xlim[1])]
    ic(X.shape, mask.shape, mask.sum())
    if isinstance(xbins, int):
        #ic(X[mask], X[mask].shape)
        #ic(X[mask].min(), X[mask].max(), np.min(X[mask]), np.max(X[mask]),
           #X.loc[mask].min(), X.loc[mask].max())
        if xscale == 'log':
            mask = mask & (X > 0)
            xbins = np.linspace(
                np.log10(np.min(X[mask])), np.log10(np.max(X[mask])), xbins+1)
        else:
            xbins = np.linspace(X[mask].min(), X[mask].max(), xbins+1)
        if xscale == 'log':
            xbins = 10**xbins
    if xscale == 'log':
        xb = np.log10(xbins)
        xcenters = 10**((xb[:-1]+xb[1:])/2)
    else:
        xcenters = (xbins[:-1]+xbins[1:]) / 2
    return xbins, xcenters


def relation_lines(x, y, xbins, statistic, mask=None, bindata=None, bins=10):
    if mask is not None:
        x = x[mask]
        y = y[mask]
        if bindata is not None:
            bindata = bindata[mask]
    ic(type(bindata), statistic)
    ic(xbins, x.min(), y.max())
    # std in dex
    if statistic == 'std':
        y = np.log10(y)
    statistic = statistic.split('/')
    if bindata is None or statistic == ['count']:
        func = binstat
        args = [x, y]
        bin_arg = xbins
    else:
        func = binstat_dd
        args = [[bindata, x], y]
        bin_arg = (bins, xbins)
    ic(x)
    ic(func.__name__)
    relation = func(*args, statistic[0], bin_arg).statistic
    if len(statistic) == 2:
        relation = relation \
             / func(*args, statistic[1], bin_arg).statistic
    # this will make the plot look nicer
    if 'std' in statistic:
        relation[relation == 0] = np.nan
    return relation


def relation_surface(x, y, xbins, ybins, statistic, mask=None,
                     bindata=None, bins=10, logbins=True, cmap='viridis'):
    ic()
    ic(bindata)
    ic(statistic)
    assert isinstance(logbins, bool)
    if mask is not None:
        x = x[mask]
        y = y[mask]
        if bindata is not None:
            bindata = bindata[mask]
    if bindata is None or statistic == 'count':
        relation = np.histogram2d(x, y, (xbins,ybins))[0]
        rel = relation[relation > 0] if logbins else relation
        vmin, vmax = np.percentile(rel, [1, 99])
        colornorm = mplcolors.LogNorm()
    elif '/' in statistic:
        stat = statistic.split('/')
        relation = binstat_dd(
                [x, y], bindata, stat[0], bins=(xbins,ybins)).statistic \
            / binstat_dd(
                [x, y], bindata, stat[1], bins=(xbins,ybins)).statistic
    else:
        relation = binstat_dd(
            [x, y], bindata, statistic, bins=(xbins,ybins)).statistic
    # maybe everything below should be a separate function
    if True:
        ic(np.percentile(relation, [0, 1, 50, 99, 100]))
        ic(vmin, vmax)
        colors, cmap = colorscale(
            array=relation, vmin=vmin, vmax=vmax, log=logbins, cmap=cmap)
    #except AssertionError as err:
        #ic(xcol, ycol, vmin, vmax)
        #ic(err)
        #sys.exit()
    if bindata is not None:
        # color scale -- still need to set vmin,vmax
        ic(colors)
        ic(relation)
        ic(relation.shape)
        colornorm = mplcolors.Normalize(vmin=vmin, vmax=vmax)
        if 'std' in statistic:
            relation[relation == 0] = np.nan
        if with_alpha:
            alpha = np.histogram2d(x, y, bins=(xbins,ybins))[0]
            ic(alpha)
            alpha_log = np.log10(alpha)
            ic(alpha_log.min(), alpha_log.max())
            alpha = alpha_log
            # should go between 0 and 1
            alpha_min = 0.4
            alpha[~np.isfinite(alpha)] = 0
            alpha = (alpha - alpha.min()) / (alpha.max()-alpha.min()) \
                + alpha_min
            alpha[alpha > 1] = 1
            ic(alpha)
            ic(alpha.shape, alpha.min(), alpha.max())
            # to add alpha
            ic(colors.shape)
            colors[:,:,-1] = alpha
            colors = np.transpose(colors, axes=(2,0,1))
            colors = colors.reshape((4,alpha.size))
            ic(colors.shape)
            cmap = ListedColormap(colors.T)
    return relation, colors, cmap


def plot_relation(sim, subs, xcol='Mstar', ycol='Mbound',
                  lines_only=True, statistic='mean', selection='Mstar',
                  selection_min=1e8, selection_max=None, xlim=None,
                  ylim=None, xbins=12, xscale='log', ybins=12, yscale='log',
                  hostmass='M200Mean', min_hostmass=13,
                  bindata=None, bincol=None, bins=6, logbins=False,
                  binlabel='', mask=None, xlabel=None, ylabel=None,
                  with_alpha=False, cmap='viridis', lw=4,
                  colornorm=mplcolors.LogNorm(), show_histograms=False,
                  show_contours=False, contour_kwargs={},
                  show_satellites=True, show_centrals=False,
                  satellites_label='All satellites',
                  centrals_label='Centrals', literature=False,
                  show_ratios=False, ylim_ratios=None, show_1to1=False):
    """Plot the SHMR and HSMR

    ``bincol`` and ``bins`` allow the relations to be binned in a
    third quantity

    If ``bincol`` is ``None`` then ``statistic`` is forced to ``count``

    """
    ic(xcol, xbins)
    ic(ycol, ybins)
    ic(bincol, logbins, bins)
    # doesn't make much sense otherwise
    if statistic != 'mean':
        show_contours = False
    count_stat = 'median'
    has_iterable_bins = np.iterable(bins)
    cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat = \
        definitions(subs, hostmass=hostmass, min_hostmass=min_hostmass)
    xdata = subs[xcol]
    ydata = subs[ycol]
    ic(xdata.shape, ydata.shape)
    mask = np.isfinite(xdata) & np.isfinite(ydata)
    ic(mask.sum())
    cmap = plt.get_cmap(cmap)
    if bincol is None:
        statistic = 'count'
        # this just for easier integration of code in the plotting bit
        bins = np.zeros(1)
        colors = 'k'
        with_alpha = False
        lines_only = False
    else:
        bindata = subs[bincol]
        mask = mask & np.isfinite(bindata)
        if not has_iterable_bins:
            if logbins:
                j = mask & (bindata > 0)
                vmin, vmax = np.log10(
                    np.percentile(bindata[j], [1,99]))
                bins = np.logspace(vmin, vmax, bins)
            else:
                vmin, vmax = np.percentile(bindata[mask], [1,99])
                bins = np.linspace(vmin, vmax, bins)
            ic(vmin, vmax)
            ic(bins)

        #ic(bincol, logbins)
        #ic(bins)
        #ic(bindata.min(), bindata.max())
        #ic(np.histogram(bindata, bins)[0])
        if not binlabel:
            binlabel = f'{statistic}({bincol})'
        if logbins:
            lb = np.log10(bins)
            bin_centers = 10**((lb[1:]+lb[:-1])/2)
        else:
            bin_centers = (bins[1:]+bins[:-1]) / 2
        ic(bin_centers)
        if has_iterable_bins:
            vmin = bin_centers[0]
            vmax = bin_centers[-1]

        #colors, colormap = colorscale(array=bin_centers, log=logbins)
        if logbins:
            normfunc = mplcolors.LogNorm
        else:
            normfunc = mplcolors.Normalize
        colormap = cm.ScalarMappable(
            normfunc(vmin=bin_centers[0], vmax=bin_centers[-1]), cmap)
        colors = colormap.to_rgba(bin_centers)
    #mask = mask & (subs[hostmass] >= 10**min_hostmass)
    # these cases should be controlled with xbins rather than selection
    if xcol == selection:
        selection = None
    if selection is not None:
        seldata = subs[selection]
        if selection_min is not None:
            mask = mask & (seldata >= selection_min)
        if selection_max is not None:
            mask = mask & (seldata <= selection_max)
    #ic(mask.sum())
    xdata = xdata[mask]
    ydata = ydata[mask]
    ic(xdata.shape, ydata.shape)
    if bincol is not None:
        bindata = bindata[mask]
    sat = sat[mask]
    dark = dark[mask]
    # cen = cen[mask]
    gsat = sat & ~dark
    ic(xbins)
    # calculate x-binning if necessary
    xbins, xcenters = do_xbins(xdata, gsat, xbins, xscale=xscale)
    logx = np.log10(xcenters)
    ic(xcol, xbins)
    ic(xdata.min(), xdata.max())
    ic(np.histogram(xdata, xbins)[0])
    ic(xcenters)
    if show_contours or not lines_only:
        ybins, ycenters = do_xbins(
            ydata, gsat, ybins, xlim=ylim if statistic == 'mean' else None,
            xscale=yscale)
        logy = np.log10(ycenters)
        ic(ycol, ybins)
        ic(ydata.min(), ydata.max())
        ic(np.histogram(ydata, ybins)[0])
        ic(ycenters)
    ic(xlim, ylim)

    mask_ = (bindata >= bins[0]) & (bindata <= bins[-1])
    relation_overall = relation_lines(
        xdata[mask_], ydata[mask_], xbins, statistic, gsat[mask_])
    ic(relation_overall)
    # at least for now
    show_ratios = show_ratios * lines_only
    if show_ratios:
        fig = plt.figure(figsize=(8,8), constrained_layout=True)
        gs = GridSpec(2, 1, height_ratios=(5,2), hspace=0.05,
                      left=0.15, right=0.9, bottom=0.1, top=0.95)
        fig.add_subplot(gs[0])
        fig.add_subplot(gs[1])#, sharex=fig.axes[0])
        #fig.add_gridspec(2, 1, height_ratios=(5,2))
        axes = fig.axes
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
        axes = [ax]
    if show_contours or show_histograms:
        counts = relation_surface(
            xdata, ydata, xbins, ybins, 'count', gsat,
            logbins=logbins, cmap=cmap)
        if show_contours:
            for key, val in zip(('zorder','color','linestyles'), (5,'k','solid')):
                contour_kwargs[key] = val
            #try:
            if 'levels' not in contour_kwargs:
                contour_kwargs['levels'] \
                    = contour_levels(
                        xdata[gsat], ydata[gsat], 12, (0.1,0.5,0.9))
            #ax.contour(xdata[gsat], ydata[gsat], contour_kwargs['levels'])
            #except: # which exception was it again?
                #pass
            #else:
            ax.contour(xcenters, ycenters, counts, **contour_kwargs)
        else:
            twinx = ax.twinx()
            twinx.hist(xdata[gsat], bins=xbins, histtype='step', color='k')
            twinx.set(yticks=[])
    if lines_only:
        ic(np.percentile(bindata, [1,50,99]))
        ic(bins)
        ic(np.histogram(bindata, bins)[0])
        relation = relation_lines(
            xdata, ydata, xbins, statistic, gsat, bindata, bins)
        ic(relation.shape)
        for i, (r, c) in enumerate(zip(relation, colors)):
            ax.plot(xcenters, r, '-', color=c, lw=4, zorder=10+i)
        if show_ratios:
            for i, (r, c) in enumerate(zip(relation, colors)):
                axes[1].plot(xcenters, r/relation_overall, '-',
                             color=c, lw=4, zorder=10+i)
            axes[1].axhline(1, ls='--', color='k', lw=1)
    else:
        relation, colors, _ = relation_surface(
            xdata, ydata, xbins, ybins, statistic, gsat, bindata, bins,
            logbins=logbins, cmap=cmap)
        xgrid, ygrid = np.meshgrid(xbins, ybins)
        ic(cmap)
        colormap = ax.pcolormesh(
            xgrid, ygrid, relation.T,# cmap=cmap_alpha if with_alpha else cmap)
            cmap=cmap, norm=colornorm, aa=False, rasterized=True)
        # see https://stackoverflow.com/questions/32177718/use-a-variable-to-set-alpha-opacity-in-a-colormap
        # apparently it is necessary to save before adding alpha
        # plt.savefig('tmp.png')
        # for i, a in zip(pcm.get_facecolors(), alpha.flatten()):
        #     i[3] = a

    ic(bins.shape, colors.shape)
    if lines_only:
        # add discrete colorbar
        cmap_lines = plt.get_cmap(cmap, bins.size-1)
        boundary_norm = mplcolors.BoundaryNorm(bins, cmap.N)
        sm = cm.ScalarMappable(cmap=cmap_lines, norm=boundary_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes, ticks=bins)
        # if ':time' in bincol:
        #     cbar.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        cbar.ax.yaxis.set_minor_locator(ticker.NullLocator())
    else:
        cbar = plt.colorbar(colormap, ax=axes)
    if statistic == 'count':
        cbar.set_label('$N$')
        if relation.max() < 1000:
            cbar.ax.yaxis.set_major_formatter(
                ticker.FormatStrFormatter('%d'))
    else:
        cbar.set_label(binlabel)
    if logbins:
        cbar.ax.set_yscale('log')
        if bins[0] >= 0.001 and bins[-1] <= 1000:
            fmt = '%d' if bins[0] >= 1 else '%s'
            cbar.ax.yaxis.set_major_formatter(
                ticker.FormatStrFormatter(fmt))
    # compare to overall and last touches
    j = mask & gsat
    nsat = binstat(xdata[j], ydata[j], 'count', xbins).statistic
    if '/' in statistic:
        st = statistic.split('/')
        satrel = binstat(xdata[j], ydata[j], st[0], xbins)[0] \
            / binstat(xdata[j], ydata[j], st[1], xbins)[0]
    else:
        st = count_stat if statistic == 'count' else statistic
        satrel = binstat(xdata[j], ydata[j], st, xbins).statistic
    if show_satellites:
        # skipping the last element because it's usually very noisy
        # (might have to check for specific combinations)
        plot_line(
            ax, xcenters[:-1], satrel[:-1], ls='-', lw=4, color=scolor,
            label=satellites_label, zorder=100)
        ic(f'{st} for all satellites:')
        ic(f'{xcol} = {xcenters}')
        ic(f'{ycol} = {satrel}')
    # centrals have not been masked
    if show_centrals:
        jcen = (subs['Rank'] == 0)
        if selection is not None:
            if selection_min is not None:
                jcen = jcen & (subs[selection] >= selection_min)
            if selection_max is not None:
                jcen = jcen & (subs[selection <= selection_max])
        if bincol is not None and 'M200Mean' not in bincol:
            if 'Distance' not in bincol and 'time' not in bincol:
                bc = '/'.join([i.split(':')[-1] for i in bincol.split('/')])
                if bc != 'z':
                    jcen = jcen & (subs[bc] >= bins[0]) & (subs[bc] <= bins[-1])
        if jcen.sum() == 0:
            show_centrals = False
        else:
            yc = '/'.join([i.split(':')[-1] for i in ycol.split('/')])
            cydata = subs[yc][jcen]
            if statistic == 'std':
                cydata = np.log10(cydata)
            if 'time' in xcol or 'Distance' in xcol \
                    or xcol.split(':')[-1] == 'z':
                ncen = -np.ones(xcenters.size, dtype=int)
            else:
                xc = '/'.join([i.split(':')[-1] for i in xcol.split('/')])
                cxdata = subs[xc][jcen]
                ncen = np.histogram(cxdata, xbins)[0]
            # in this case just show overall mean
            if 'time' in xcol or ('Distance' in xcol and '/' in ycol) \
                    or xcol.split(':')[-1] == 'z':
                if statistic == 'mean':
                    c0 = np.mean(cydata)
                elif statistic == 'std':
                    c0 = np.std(cydata)
                else:
                    c0 = np.std(cydata) / np.mean(cydata)
                cenrel = c0 * np.ones(xcenters.size)
            elif '/' in statistic:
                st = statistic.split('/')
                cenrel = binstat(cxdata, cydata, st[0], xbins)[0] \
                    / binstat(cxdata, cydata, st[1], xbins)[0]
            else:
                st = count_stat if statistic == 'count' else statistic
                cenrel = binstat(cxdata, cydata, st, xbins)[0]
            cenrel[cenrel == 0] = np.nan
            ic(cenrel)
            ls = '--' if 'Distance' in xcol else 'o--'
            plot_line(
                ax, xcenters, cenrel, ls='o--', lw=2, color=ccolor, ms=6,
                label=centrals_label, zorder=100)
    else:
        cenrel = -np.ones(xcenters.size)
        ncen = np.zeros(xcenters.size)

    if literature:
        xcol_split = xcol.split(':')
        if len(xcol_split) <= 3 and xcol_split[-1] == 'Mstar':
            xlit = 10**np.array([9.51, 10.01, 10.36, 10.67, 11.01])
            #ylit = read_literature('sifon18_mstar', 'Msat_rbg')
        elif 'Distance' in xcol:
            # Rsat (Mpc) - missing normalization
            xlit = np.array([0.23, 0.52, 0.90, 1.55])
            #ylit = read_literature('sifon18_Rbcg', 'Msat_rbg')
    if literature or show_centrals:
        ax.legend(fontsize=18)
    if xlabel is None:
        #xlabel = r'$\log\,{0}$'.format(sim.masslabel(mtype='stars'))
        xlabel = get_axlabel(xcol, statistic)
    if ylabel is None:
        #ylabel = r'$\log\,{0}$'.format(sim.masslabel(mtype='total'))
        ylabel = get_axlabel(ycol, statistic)
    # cheap hack
    xlabel = xlabel.replace('$$', '$')
    ylabel = ylabel.replace('$$', '$')
    #if xlim is None:
        #xlim = np.transpose([ax.get_xlim() for ax in axes])
        #xlim = (np.min(xlim[0]), np.max(xlim[1]))
    #ic(xlim)
    for ax in axes:
        ax.set(xscale=xscale)
        #ax.set_xlim(xlim)
    axes[0].set(ylabel=ylabel, yscale=yscale)
    if ylim is None:
        if statistic == 'std':
            ylim = (0, 1.2)
        elif statistic == 'std/mean':
            ylim = (0, 1.5)
    if ylim is not None:
        axes[0].set_ylim(ylim)
    axes[-1].set(xlabel=xlabel)
    if show_ratios:
        axes[0].set(xticklabels=[])
        axes[1].set(ylabel='Ratios')
    if 'Distance' in xcol and xscale == 'log':
        axes[-1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%s'))
    elif 'time' in xcol:
        for ax in axes:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        axes[-1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    if yscale == 'log':
        ylim = axes[0].get_ylim()
        ic(ylim)
        axes[0].yaxis.set_minor_formatter(ticker.NullFormatter())
        if ylim[0] >= 0.001 and ylim[1] <= 1000:
            fmt = '%d' if ylim[0] >= 1 else '%s'
            axes[0].yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
        if show_ratios:
            if ylim_ratios is not None:
                axes[1].set_ylim(ylim_ratios)
            ylim = axes[1].get_ylim()
            ic(ylim)
            ic(ylim[0] > 0 and ylim[1]/ylim[0] >= 50,
               ylim[0] <= 0 and ylim[1] >= 20)
            if (ylim[0] > 0 and ylim[1]/ylim[0] >= 50) \
                    or (ylim[0] <= 0 and ylim[1] >= 20):
                axes[1].set_yscale('log')
                axes[1].yaxis.set_major_formatter(
                    ticker.FormatStrFormatter('%s'))
            #axes[1].set_yticks(np.logspace(, 4, 9))
            #axes[1].set_ylim(ylim)

    # diagonal line when applicable (we do this after setting the limits)
    if show_1to1:
        xlim = axes[0].get_xlim()
        ylim = axes[0].get_ylim()
        d0 = min([xlim[0], ylim[0]])
        d1 = max([xlim[1], ylim[1]])
        axes[0].plot([d0, d1], [d0, d1], 'k--')

    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, hspace=0.05)
    ic(axes[0].get_xlim())
    ic(axes[0].get_ylim())
    # add a random number to see whether they update
    # ax.annotate(
    #     str(np.random.randint(1000)), xy=(0.96,0.96), xycoords='axes fraction',
    #     fontsize=14, va='top', ha='right', color='C3')
    # format filename
    xcol = xcol.replace('-', '-minus-').replace('/', '-over-').replace(':', '-')
    ycol = ycol.replace('-', '-minus-').replace('/', '-over-').replace(':', '-')
    statistic = statistic.replace('/', '-over-')
    outcols = f'{xcol}_{ycol}'
    outdir = f'{statistic}__{outcols}'
    if bincol is None:
        outname = outdir
    else:
        bincol = bincol.replace('-', '-minus-').replace('/', '-over-').replace(':', '-')
        outname = f'{outdir}__bin__{bincol}'
    if show_centrals:
        outname = f'{outname}__withcen'
    if show_satellites:
        outname = f'{outname}__withsat'
    if show_ratios:
        outname = f'{outname}__ratios'
    root = 'relations/lines' if lines_only else 'relations/surface'
    output = os.path.join(root, ycol, outcols, outdir, outname)
    output = save_plot(fig, output, sim, tight=False)
    txt = output.replace('.pdf', '.txt')
    # note that this is only the mean relation, not binned by bincol
    #np.savetxt(txt, np.transpose([xcenters, nsat, satrel, ncen, cenrel]),
               #fmt='%.5e', header=f'{xcol} Nsat {ycol}__sat Ncen {ycol}__cen')
    np.savetxt(txt, np.vstack([xcenters, relation]), fmt='%.5e')
    return relation
