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

from plottery.plotutils import colorscale, savefig, update_rcParams
from plottery.statsplots import contour_levels
update_rcParams()

from astro.clusters.conversions import rsph
from lnr import to_linear, to_log

from ..hbt_tools import format_colname, save_plot
#from hbtpy.hbtplots import RelationPlotter
#from hbtpy.hbtplots.core import ColumnLabel
from .plot_auxiliaries import (
    binning, definitions, get_axlabel, get_bins, get_label_bincol, logcenters,
    massbins, plot_line)
from .plot_definitions import (
    ccolor, scolor, massnames, units, xbins, binlabel, events, axlabel, ylims)


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
            min_hostmass=logM200Mean_min, show_ratios=False
            #, xlim=(3e7,1e12))
            )

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
    if 'yscale' not in kwargs:
        kwargs['yscale'] = 'log' if statistic == 'mean' else 'linear'
    if statistic not in ('count', 'mean') \
            and ('ylim' not in kwargs or kwargs['ylim'] is None):
        kwargs['ylim'] = (0, 0.6) if statistic == 'std' else (0, 1.5)
    if 'xbins' in kwargs:
        x_bins = kwargs.pop('xbins')
    if statistic == 'count':
        kwargs['show_centrals'] = False
        show_satellites = False
    plot_relation(
        sim, subs, xcol=xcol, ycol=ycol, xbins=x_bins, statistic=statistic,
        xscale=xscale, bincol=bincol, binlabel=label, bins=bins,
        logbins=logbins, xlabel=get_axlabel(xcol, 'mean'),
        ylabel=get_axlabel(ycol, statistic),
        show_satellites=show_satellites, **kwargs)
        #selection=selection, selection_min=selection_min,
        #show_satellites=show_satellites, show_centrals=show_centrals)
    return


def wrap_relations_distance(args, stat):
    xscale = 'log'
    xb = np.logspace(-2, 0.5, 8) if xscale == 'log' \
        else np.linspace(0, 3, 8)
    d = 'ComovingMostBoundDistance'
    plot_args = []
    # defined here to avoid repetition
    ycols_present = ['Mbound/Mstar', 'Mstar/Mbound', 'Mstar', 'Mbound',
                     'Mbound/M200Mean']
    bincols_present = ['M200Mean', 'Mbound', 'Mstar']
    for ie, event in enumerate(('first_infall', 'last_infall', 'cent', 'sat')):
        h = f'history:{event}'
        ycols = [f'Mbound/{h}:Mbound', f'Mstar/{h}:Mstar',
                 f'{h}:time', f'{h}:Mbound/{h}:Mstar',
                 f'{h}:Mbound', f'{h}:Mstar'] \
                + ycols_present
        bincols = [f'{h}:time', f'{h}:Mstar', f'{h}:Mbound',
                   f'{h}:Mbound/{h}:Mstar'] \
                  + bincols_present
        # ycols = ['Mbound/Mstar']
        # bincols = ['Mstar']
        # uncomment when testing
        # ycols = ['Mbound']
        # bincols = ['Mstar']
        for ycol in ycols:
            if stat == 'mean' \
                    and ycol in ('Mbound/Mstar', f'{h}:Mbound/{h}:Mstar'):
                ylim = (2, 100)
            else: ylim = None
            #ylim = None
            for xyz in ('0', ''):
                for xcol in (f'{d}{xyz}/R200MeanComoving', f'{d}{xyz}'):
                #for xcol in (f'{d}{xyz}/R200MeanComoving',):
                    # 2d histogram
                    if stat == 'count':
                        kwds = dict(
                            x_bins=np.logspace(-2, 0.7, 26), ybins=20,
                            selection='Mstar', selection_min=1e8,
                            xscale='log', bins=None, statistic=stat,
                            lines_only=False, ylim=ylims.get(ycol))
                        plot_args.append([[xcol, ycol, None], kwds.copy()])
                    # line plots
                    else:
                        lit = (len(xyz) > 0) and stat == 'mean'
                        kwds = dict(
                            x_bins=xb, xscale=xscale, literature=lit,
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
                            if bincol == 'Mstar': kwds['bins'] = np.logspace(9, 11, 5)
                            else: kwds['bins'] = get_bins(bincol, logbins, n=5)
                            if ycol == 'Mbound/Mstar' and bincol == 'M200Mean':
                                kwds['ylim_ratios'] = (0.5, 1.5)
                            else:
                                kwds['ylim_ratios'] = None
                            plot_args.append(
                                [[xcol, ycol, bincol], kwds.copy()])
        #             break
        #         break
        #     break
        # break
    return plot_args


def wrap_relations_hsmr(args, stat, **relation_kwargs):
    x_bins = {'Mstar': np.logspace(7.5, 11.7, 10),
              'Mbound': np.logspace(8.5, 12.7, 10)}
    events = [f'history:{e}'
              for e in ('first_infall', 'last_infall', 'cent', 'sat',
                        'max_Mstar', 'max_Mbound', 'max_Mdm')]
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
    #bincols = ['M200Mean']
    #bincols = ['Mstar/history:last_infall:Mbound']
    # bincols = ['ComovingMostBoundDistance/R200MeanComoving',
    #            'ComovingMostBoundDistance0/R200MeanComoving']
    #bincols = ['M200Mean', 'history:sat:time']
    #ic(bincols)
    kwds_count = dict(
        x_bins=20, ybins=20, selection='Mstar', selection_min=1e8,
        force_selection=True, statistic='count', lines_only=False,
        show_satellites=True)
    #for ycol in ('Mbound', 'Mbound/Mstar', 'Mtotal', 'Mtotal/Mstar'):
    plot_args = []
    #ycols = ['NboundType4']
    #ylims = [None]
    ycols = ('Mbound/Mstar', 'Mstar/Mbound', 'Mbound', 'M200Mean')
    ylims = ((3,200), (2e-3,0.5), (5e8,2e14), (1e13,5e14))
    for ycol, ylim in zip(ycols[:3], ylims):
        # histograms
        if stat == 'count':
            kwds = {**kwds_count, **dict(ylim=ylim)}
            kwds['yscale'] = 'linear' if '/' in ycol else 'log'
            plot_args.append([['Mstar', ycol, None], kwds.copy()])
        # lines
        else:
            for bincol in bincols:
                #if bincol != 'M200Mean': continue
                logbins = ('time' not in bincol)
                bins = get_bins(bincol, logbins, n=4+('time' in bincol))
                kwds = {**dict(bins=bins, statistic=stat,
                               show_ratios=False,
                               show_satellites=False, show_centrals=True),
                        **relation_kwargs}
                kwds['ylim'] = ylim if stat == 'mean' else None
                for xcol in ('Mstar', 'Mbound'):
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
    plot_args.append(
        [['Mstar/history:first_infall:Mbound',
          'Mbound/history:first_infall:Mbound', None],
         dict(statistic='count')])
    events = ('first_infall', 'last_infall', 'cent', 'sat', 'max_Mstar',
              'max_Mbound')
    events = ('max_Mbound',)
    for ie, event in enumerate(events):
        h = f'history:{event}'
        # ratios in the x-axis
        if do_ratios:
            xcols = [f'{m}/{h}:{m}' for m in ('Mbound','Mdm','Mgas','Mstar')] \
                + [f'{h}:{m}/{h}:Mbound' for m in ('Mdm','Mgas','Mstar')] \
                + [f'{m}:history:max_{m}:{m}' for m in ('Mbound','Mdm','Mgas','Mstar')]
            ycols = [f'{m}/{h}:{m}' for m in ('Mbound','Mdm','Mgas','Mstar')] \
                + [f'{h}:{m}/{h}:Mbound' for m in ('Mdm','Mgas','Mstar')] \
                + [f'{m}:history:max_{m}:{m}' for m in ('Mbound','Mdm','Mgas','Mstar')]
            ic(np.sort(xcols))
            ic(np.sort(ycols))
            bincols = [f'{h}:z', f'{h}:time', 'M200Mean', 'Mbound/M200Mean',
                       'Mstar', f'{h}:Mstar'] \
                + [f'history:{e}:time-{h}:time' for e in events if e != event]
            kwds['show_centrals'] = False
            kwds['show_1to1'] = True
            kwds['yscale'] = 'log'
            kwds['xbins'] = 25
            kwds['ybins'] = 25
            for xc in xcols:
                for yc in ycols:
                    if 'max_Mstar' not in xc and 'max_Mdm' not in yc:
                        continue
                    if xc == yc:
                        continue
                    if stat == 'count':
                        plot_args.append([[xc, yc, None], kwds.copy()])
                        continue
                    for bincol in bincols:
                        if xc == bincol or yc == bincol: continue
                        logbins = ('time' not in bincol)
                        ic(bincol, logbins)
                        bins = get_bins(bincol, logbins, n=5)
                        kwds['logbins'] = logbins
                        plot_args.append([[xc, yc, bincol], kwds.copy()])
                        # for xc, yc, xb in zip(xcol, ycol, x_bins):
                        #     kwds['xlim'] = (xb[0], xb[-1])
                        #     kwds['ylim'] = (xb[0], xb[-1])
                        #     plot_args.append([[xc, yc, bincol, xb], kwds.copy()])
        # mass in the x-axis
        if do_mass:
            bincols = (f'{h}:z', f'{h}:time',
                       'Mstar', 'Mbound', 'Mstar/Mbound', 'Mbound/Mstar',
                       'Mdm/Mstar', 'LastMaxMass', 'Mbound/LastMaxMass',
                       'Mstar/LastMaxMass', 'M200Mean', 'Mbound/M200Mean',
                       'ComovingMostBoundDistance','ComovingMostBoundDistance0',
                       'ComovingMostBoundDistance/R200Mean',
                       'ComovingMostBoundDistance0/R200Mean')
            for bincol in bincols:
                if 'time' not in bincol: continue
                logbins = ('time' not in bincol \
                    and bincol.split(':')[-1] != 'z')
                ic(bincol, logbins)
                bins = get_bins(bincol, logbins,
                                n=None if bincol.split(':')[-1] == 'z' else 5)
                kwds = {**dict(bins=bins, statistic=stat, logbins=logbins),
                        **relation_kwargs}
                ycols = [f'{h}:Mbound', f'{h}:Mbound/{h}:Mstar',
                         f'Mbound/{h}:Mbound', f'Mstar/{h}:Mstar']
                ylims = [(1e9, 1e13), (3, 300), (0.01, 2), (0.01, 2)]
                ycols = ycols \
                    + [f'{h}:{m}' for m in ('Mbound','Mdm','Mgas','Mstar')]
                ylims = ylims + [(1e8, 1e13), (1e8, 1e13), (1e7, 1e11), (1e7, 1e12)]
                ycols = [ycols[1]]
                ylims = [ylims[1]]
                if stat == 'count':
                    for ycol in ycols:
                        #for xcol in ('Mstar', f'{h}:Mstar'):
                        for xcol in (f'{h}:Mstar', f'{h}:Mbound'):
                            if xcol == ycol:
                                continue
                            kwds['xbins'] = 25
                            kwds['ybins'] = 25
                            plot_args.append([[xcol, ycol, None], kwds.copy()])
                            kwds.pop('xbins')
                            kwds.pop('ybins')
                else:
                    for ycol, ylim in zip(ycols, ylims):
                    #for ycol in ycols
                        #else:
                        for xcol in (f'{h}:Mstar',):# f'{h}:Mbound'):
                            if 'history' in xcol: xb = np.logspace(9, 11.5, 11)
                            xb = xbins[xcol.split(':')[-1]]
                            kwds['ylim'] = ylim if stat == 'mean' else (0, 0.6)
                            plot_args.append(
                                [[xcol, ycol, bincol, xb], kwds.copy()])
                continue
                #
                x_bins = np.logspace(9, 13, 10)
                kwds['xlim'] = (x_bins[0], x_bins[-1])
                for ycol in ('Mbound', 'Mstar'):
                    plot_args.append(
                        [[f'{h}:Mbound', ycol, bincol, x_bins], kwds.copy()])
                #break
                # a few additional ones
                # plot_args.append(
                #     [['history:sat:Mstar', 'history:first_infall:Mstar',
                #     np.logspace(9.5, 12.5, 7)], kwds.copy()])
                # plot_args.append(
                #     [['history:sat:Mbound', 'history:first_infall:Mbound',
                #     np.logspace(10, 13, 7)], kwds.copy()])
    return plot_args


def wrap_relations_time(args, stat, do_time_differences=False):
    #x_bins = np.arange(0, 14, 1)
    plot_args = []
    kwds_count = dict(
        x_bins=30, ybins=30, selection='Mstar', selection_min=1e8,
        xscale='linear', statistic='count', yscale='log',
        show_satellites=True, lines_only=False)
    kwds = dict(
        x_bins=xbins['time'], logbins=True, selection='Mstar',
        statistic=stat, xscale='linear', yscale='log', selection_min=1e8,
        show_satellites=False, show_ratios=False)
    events = ('max_Mstar', 'max_Mbound', 'max_Mdm', 'sat', 'cent',
              'first_infall', 'last_infall')
    for ie, event in enumerate(events):
    #for event in ('max_Mbound', 'max_Mstar'):
        #if 'first' not in event: continue
        h = f'history:{event}'
        xcols = [f'{h}:z', f'{h}:time', 'history:first_infall:time'] \
            + [f'history:{e}:time-{h}:time' for e in events if e != event]
        for xcol in xcols:
            if 'time' not in xcol: continue
        #for xcol in (f'{h}:time',):
        #xcol = f'{h}:time'
            # histograms
            if stat == 'count':
                # ycols = [f'{h}:Mbound/{h}:Mstar']
                # ycols = [f'{h}:time',
                #          f'history:first_infall:Mbound/{h}:Mbound',
                #          f'history:first_infall:Mdm/{h}:Mdm',
                #          f'history:first_infall:Mstar/{h}:Mstar',
                #          f'{h}:Mbound/LastMaxMass',
                #          'ComovingMostBoundDistance/R200MeanComoving',
                #          'ComovingMostBoundDistance0/R200MeanComoving']
                # for ycol in ('Mbound', 'Mbound/Mstar', f'{h}:Mbound',
                #              f'Mbound/{h}:Mbound', f'Mstar/{h}:Mbound',
                #              f'Mstar/{h}:Mstar', 'M200Mean', 'Mbound/M200Mean',
                #              f'Mdm/{h}:Mdm', f'Mgas/{h}:Mgas',
                #              f'Mstar/Mbound', f'Mdm/Mbound', f'Mgas/Mbound',
                #              'ComovingMostBoundDistance/R200MeanComoving'):
                ycols = [f'{h}:Mstar', f'Mstar/{h}:Mstar', 'Mstar',
                         f'{h}:Mbound', f'Mbound/{h}:Mbound',
                         f'{h}:Mbound/{h}:Mstar', 'Mbound'] \
                    + [f'history:{e}:time-{h}:time' for e in events if e != event]
                #ycols = [f'{h}:Mbound']
                if 'time' in xcol:
                    ycols = ycols \
                        + [f'history:{e2}:time-{h}:time'
                           for e2 in events if e2 != event]
                for ycol in ycols:
                    kwds_count['ylim'] = ylims.get(ycol)
                    kwds_count['yscale'] \
                        = 'linear' if ('Distance' in ycol or 'time' in ycol) \
                            else 'log'
                    kwds_count['ybins'] \
                        = np.linspace(0, 1.5, 15) if 'Distance' in ycol else 20
                    plot_args.append([[xcol, ycol, None], kwds_count.copy()])
                if do_time_differences:
                    for i, e in enumerate(events):
                        if i == ie: continue
                        hi = f'history:{e}'
                        ycols = (f'{hi}:time', f'{hi}:time-{xcol}')
                        for ycol in ycols:
                            if xcol == ycol: continue
                            kwds_count['ylim'] = ylims.get(ycol)
                            kwds_count['yscale'] \
                                = 'linear' if ('Distance' in ycol
                                               or 'time' in ycol) \
                                    else 'log'
                            kwds_count['ybins'] \
                                = np.linspace(0, 1.5, 15) \
                                    if 'Distance' in ycol else 20
                            plot_args.append(
                                [[xcol, ycol, None], kwds_count.copy()])
            # lines
            else:
                #for ycol in (f'{h}:M200Mean',):
                ycols = ['Mbound/M200Mean',
                         'ComovingMostBoundDistance/R200MeanComoving',
                         'ComovingMostBoundDistance0/R200MeanComoving'] \
                    + [f'{m}/{h}:{m}' for m in ('Mbound','Mdm','Mgas','Mstar')] \
                    + [f'{h}:{m}/{h}:Mbound' for m in ('Mdm','Mgas','Mstar')]
                ycols = ycols + [f'{h}:Mstar/history:max_Mstar:Mstar',
                         f'{h}:Mbound/history:max_Mbound:Mbound',
                         f'{h}:time-history:max_Mstar:time']
                    # for bincol in (f'{h}:Mbound/{h}:Mstar',
                    #                f'{h}:Mstar/{h}:Mbound',
                    #                'M200Mean', 'Mbound/Mstar', 'Mstar/Mbound',
                    #                'Mstar', 'Mbound',
                    #                f'{h}:Mbound', f'{h}:Mstar'):
                #for ycol in (f'Mdm/{h}:Mdm', f'Mstar/{h}:Mstar'):
                ycols = ycols + [f'history:first_infall:Mbound/{h}:Mbound',
                         f'history:first_infall:Mdm/{h}:Mdm',
                         f'history:first_infall:Mgas/{h}:Mgas',
                         f'history:first_infall:Mstar/{h}:Mstar']
                    #for bincol in (f'{h}:Mbound',):
                                   #f'{h}:time-history:first_infall:time'):
                # for m in ('Mbound','Mdm','Mstar','Mgas'):
                #     hm = f'history:max_{m}'
                #     ycol = f'{hm}:{m}'
                #     kwds['show_centrals'] = not (h in ycol or hm in ycol)
                bincols = [f'{h}:Mbound', f'{h}:Mbound/{h}:Mstar',
                           'Mstar', f'{h}:Mstar', f'{h}:Mbound/{h}:M200Mean',
                           f'{h}:time', f'{h}:Mgas', f'{h}:Mgas/{h}:Mstar',
                           f'{h}:Mgas/{h}:Mbound',
                           f'Mstar/{h}:Mstar', 'M200Mean'] \
                    + [f'history:{e}:time-{h}:time' for e in events if e != events]
                # bincols = [f'{h}:Mgas', f'{h}:Mgas/{h}:Mbound',
                #            f'{h}:Mgas/{h}:Mstar']
                #yc = bincols[]
                for ycol in ycols:
                    for bincol in bincols:
                        if bincol == ycol or bincol in xcol:
                            continue
                        if bincol == f'{h}:Mbound':
                            bins = np.logspace(10.5, 12.5, 5)
                        else:
                            bins = np.logspace(9, 11.3, 5) \
                                if ('Mstar' in bincol and '/' not in bincol) \
                                else get_bins(bincol, n=4)
                        kwds['yscale'] = 'linear' \
                            if ycol in (f'Mstar/{h}:Mstar', f'Mdm/{h}:Mdm') \
                            else 'log'
                        kwds['yscale'] = 'linear'
                        kwds['ylim'] = (0, 1) if ycol == f'Mdm/{h}:Mdm' \
                            else None
                        #kwds['ylim'] = (0.5, 2)
                        kwds['bins'] = bins
                        kwds['logbins'] = ('time' not in bincol)
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


def plot_segregation_literature(ax, xcol, ycol):
    ## Sifón+18
    # Rsat (Mpc) - missing normalization
    xlit = np.array([0.23, 0.52, 0.90, 1.55])
    if '/R200Mean' in xcol:
        xlit /= 2.36
    logylit = [10.49, 11.60, 11.55, 11.46]
    ylit, ylitlo = to_linear(
        logylit, [0.47, 0.15, 0.21, 0.33], which='lower')
    ylit, ylithi = to_linear(
        logylit, [0.35, 0.16, 0.21, 0.25], which='upper')
    if ycol == 'Mbound/Mstar':
        mstarlit = 10**np.array([9.97, 10.03, 10.07, 10.24]) 
        ylit /= mstarlit
        ylitlo /= mstarlit
        ylithi /= mstarlit
    ax.errorbar(
        xlit, ylit, (ylitlo,ylithi), fmt='o', ms=10, elinewidth=3,
        color='k', zorder=100,
        label=r'Sifón+18 ($\log\langle' \
            ' m_{\u2605}/\mathrm{M}_\odot' \
            r' \rangle=10.1$)')
    # Kumar+22
    xlit = np.array([0.2, 0.38, 0.58, 0.78])
    if '/R200Mean' in xcol:
        mcl_kumar = 10**np.array([14.31, 14.33, 14.36, 14.39])
        r200m_kumar = rsph(mcl_kumar, 0.26, ref='200m')
        xlit /= r200m_kumar
    logylit = [11.86, 12.17, 12.05, 12.11]
    ylit, ylitlo = to_linear(
        logylit, [0.20, 0.06, 0.06, 0.05], which ='lower')
    ylit, ylithi = to_linear(
        logylit, [0.16, 0.05, 0.06, 0.05], which='upper')
    if ycol == 'Mbound/Mstar':
        mstarlit = 10**np.array([10.48, 10.46, 10.50, 10.51])
        ylit /= mstarlit
        ylitlo /= mstarlit
        ylithi /= mstarlit
    ax.errorbar(
        xlit, ylit, (ylitlo,ylithi), fmt='s', ms=8, elinewidth=3,
        color='k', mfc='w', mew=3, zorder=100,
        label=r'Kumar+22 ($\log\langle' \
            ' m_{\u2605}/\mathrm{M}_\odot' \
            r' \rangle=10.5$)')
    return


def relation_lines(x, y, xbins, statistic, mask=None, bindata=None, bins=10,
                   return_err=False):
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
    ic(func)
    ic(statistic)
    if bindata is not None:
        ic(bindata.shape)
    ic(x.shape, y.shape)
    relation = func(*args, statistic[0], bin_arg).statistic
    if len(statistic) == 2:
        relation = relation \
             / func(*args, statistic[1], bin_arg).statistic
    ic(relation.shape)
    # this will make the plot look nicer
    if 'std' in statistic:
        relation[relation == 0] = np.nan
    if return_err:
        # this may not work for ratio statistics
        errlo = func(
            *args, lambda x: (np.mean(x)-np.percentile(x, 16))/x.size**0.5,
            bin_arg).statistic
        errhi = func(
            *args, lambda x: (np.percentile(x, 84)-np.mean(x))/x.size**0.5,
            bin_arg).statistic
        ic(errlo.shape, errhi.shape)
        if statistic == 'std':
            errlo /= 2**0.5
            errhi /= 2**0.5
        return relation, (relation-errlo, errhi+relation)
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
        vmin, vmax = np.percentile(relation, [1, 99])
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
                  selection_min=1e8, selection_max=None,
                  force_selection=False, xlim=None,
                  ylim=None, xbins=12, xscale='log', ybins=12, yscale='log',
                  hostmass='M200Mean', min_hostmass=13, show_hist=False,
                  bindata=None, bincol=None, bins=6, logbins=False,
                  binlabel='', mask=None, xlabel=None, ylabel=None,
                  with_alpha=False, cmap='inferno', lw=4,
                  colornorm=mplcolors.LogNorm(), cmap_range=(0.2,0.7),
                  show_contours=True, contour_kwargs={},
                  show_uncertainties=True, alpha_uncertainties=0.25,
                  show_satellites=True, show_centrals=False,
                  show_satellites_err=False, show_centrals_scatter=False,
                  satellites_label='All satellites',
                  centrals_label='Centrals', literature=False,
                  show_ratios=False, ylim_ratios=None, show_1to1=False):
    """Plot the relation between two quantities, optionally
    binning by a third

    ``bincol`` and ``bins`` allow the relations to be binned in a
    third quantity

    If ``bincol`` is ``None`` then ``statistic`` is forced to ``count``

    """
    ic(xcol, xscale, xbins)
    ic(ycol, yscale, ybins)
    ic(bincol, logbins, bins)
    # doesn't make much sense otherwise
    if statistic != 'mean':
        show_contours = False
    count_stat = np.nanmedian
    statmap = {'mean': np.nanmean, 'median': np.nanmedian, 'std': np.nanstd,
               'std/mean': lambda x: np.std(x)/np.nanmean(x)}
    has_iterable_bins = np.iterable(bins)
    cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat = \
        definitions(subs, hostmass=hostmass, min_hostmass=min_hostmass)
    xdata = subs[xcol]
    ydata = subs[ycol]
    ic(xdata.shape, ydata.shape)
    mask = np.isfinite(xdata) & np.isfinite(ydata)
    ic(mask.sum())
    #cmap = plt.get_cmap(cmap)
    cmap = cmr.get_sub_cmap(cmap, *cmap_range)
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
                ic(bindata.min(), bindata.max(), j.sum())
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
    if xcol == selection and not force_selection:
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

    if bindata is None:
        mask_ = np.ones(xdata.size, dtype=bool)
    else:
        mask_ = (bindata >= bins[0]) & (bindata <= bins[-1])
    relation_overall = relation_lines(
        xdata[mask_], ydata[mask_], xbins, statistic, gsat[mask_])
    ic(relation_overall)
    # uncertainties on overall relation
    if statistic in ('mean', 'std'):
        ic(gsat.shape, mask_.shape)
        relation_overall_std = relation_lines(
            xdata[mask_], ydata[mask_], xbins, 'std', gsat[mask_])
        relation_overall_cts = np.histogram(xdata[mask_ & gsat], xbins)[0]
        relation_overall_err = relation_overall_std / relation_overall_cts**0.5
        if statistic == 'std':
            relation_overall_err /= 2**0.5
        ic(relation_overall_err)
        #return
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
    if show_contours:
        for key, val in zip(('zorder','color','linestyles'), (5,'k','solid')):
            contour_kwargs[key] = val
        try:
            if 'levels' not in contour_kwargs:
                contour_kwargs['levels'] \
                    = contour_levels(
                        xdata[gsat], ydata[gsat], 12, (0.1,0.5,0.9))
            ax.contour(xdata[gsat], ydata[gsat], contour_kwargs['levels'])
        except: # which exception was it again?
            pass
        else:
            counts = relation_surface(
                xdata, ydata, xbins, ybins, 'count', gsat,
                logbins=logbins, cmap=cmap)
            plt.contour(xcenters, ycenters, counts, **contour_kwargs)
    if lines_only:
        ic(np.percentile(bindata, [1,50,99]))
        ic(bins, np.histogram(bindata, bins)[0])
        relation = relation_lines(
            xdata, ydata, xbins, statistic, gsat, bindata, bins,
            return_err=show_uncertainties)
        if show_uncertainties:
            relation, (err_lo, err_hi) = relation
            ic(err_lo[0], err_hi[0])
        ic(relation.shape)
        for i, (r, c) in enumerate(zip(relation, colors)):
            ax.plot(xcenters, r, '-', color=c, lw=4, zorder=10+i)
            if show_uncertainties:
                ax.fill_between(
                    xcenters, err_lo[i], err_hi[i], color=c, zorder=-10+i,
                    alpha=0.4, lw=0)
        # highlight the bin around logmstar=10 for comparison with
        # observations
        # if literature and 'Distance' in xcol and bincol == 'Mstar':
        #     ax.plot(xcenters, relation[2], '-', color=colors[2],
        #             lw=8, zorder=13)
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
        ic(cmap.name)
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
        try:
            cmap_lines = plt.get_cmap(cmap, bins.size-1)
        except AttributeError:
            cmap_lines = cmr.take_cmap_colors(cmap, bins.size-1, cmap_range=cmap_range)
        boundary_norm = mplcolors.BoundaryNorm(bins, cmap.N)
        sm = cm.ScalarMappable(cmap=cmap_lines, norm=boundary_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes, ticks=bins)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        if 'time' in bincol:
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
        s_ = np.s_[:-1]
        plot_line(
            ax, xcenters[s_], satrel[s_], ls='-', lw=4, color=scolor,
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
                try:
                    st = getattr(np, f'nan{st}')
                except AttributeError:
                    pass
                cenrel = binstat(cxdata, cydata, st, xbins)[0]
            cenrel[cenrel == 0] = np.nan
            # specifically historical mbound/mstar vs mstar binned by time
            if 'history' in xcol and 'Mstar' in xcol and bincol.count('time') == 1 \
                    and 'Mbound' in ycol and 'Mstar' in ycol \
                    and ycol.count('history') == 2:
                f = os.path.join(sim.data_path, f'chsmr_{statistic}.txt')
                print(f, os.path.isfile(f))
                cen_file = os.path.join(sim.data_path, f'chsmr_{statistic}.txt')
                cen_time = np.loadtxt(cen_file)
                print(cen_time.shape)
                for tref, ls in zip((0, 5, 10), ('o--', 's-.', '^:')):
                    # load stellar masss bins
                    with open(cen_file) as cf:
                        xcen = np.array(
                            [cf.readline() for i in range(3)][-1][2:].split(','),
                            dtype=float)
                    xcen = (xcen[:-1]+xcen[1:]) / 2
                    jcentime = np.argmin(np.abs(cen_time[:,1]-tref))
                    ic(tref, jcentime)
                    ycentime = cen_time[jcentime,3:]
                    ycentime[ycentime == 0] = np.nan
                    ic(ycentime)
                    jjcen = (xcen <= xcenters[-1])
                    # plot_line(
                    #     ax, xcen[jjcen], ycentime[jjcen], ls=ls, lw=2,
                    #     color='C3', ms=4, label=f'Centrals {tref} Gyr ago')
            #else:
            ls = '--' if 'Distance' in xcol else 'o--'
            ic(cenrel)
            plot_line(
                ax, xcenters, cenrel, ls='o--', lw=2, color=ccolor, ms=6,
                label=centrals_label, zorder=100)
    else:
        cenrel = -np.ones(xcenters.size)
        ncen = np.zeros(xcenters.size)
    if 'Distance' in xcol and 'Mbound' in ycol \
            and ('Mstar' in ycol or ycol.count('Mbound') == 2) \
            and (xcol.split('/')[0][-1] not in '012') \
            and statistic == 'mean':
        t = np.logspace(-1.6, -0.2, 10)
        ax.plot(t, 100*t**(2/3), '-', color='0.5', lw=2)
        ax.text(0.1, 30, r"$\propto (R/R_\mathrm{200m})^{2/3}$", va='center',
                ha='center', color='0.5', rotation=45, fontsize=14)

    if literature:
        xcol_split = xcol.split(':')
        if len(xcol_split) <= 3 and xcol_split[-1] == 'Mstar':
            xlit = 10**np.array([9.51, 10.01, 10.36, 10.67, 11.01])
            #ylit = read_literature('sifon18_mstar', 'Msat_rbg')
        elif 'Distance' in xcol and xcol.split('/')[0][-1] in '012' \
                and 'Mstar' in ycol:
            plot_segregation_literature(ax, xcol, ycol)
    if literature or show_centrals:
        ax.legend(fontsize=14, loc='upper left')
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
        # for ax in axes:
        #     ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
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
        axes[0].set(xlim=xlim, ylim=ylim)

    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, hspace=0.05)
    ic(axes[0].get_xlim())
    ic(axes[0].get_ylim())
    # add a random number to see whether they update
    # ax.annotate(
    #     str(np.random.randint(1000)), xy=(0.96,0.96), xycoords='axes fraction',
    #     fontsize=14, va='top', ha='right', color='C3')
    # format filename
    xcol = format_colname(xcol)
    ycol = format_colname(ycol)
    statistic = statistic.replace('/', '-over-')
    outcols = f'{xcol}__{ycol}'
    outdir = f'{statistic}__{outcols}'
    if bincol is None:
        outname = outdir
    else:
        bincol = format_colname(bincol)
        outname = f'{outdir}__bin__{bincol}'
    if show_centrals:
        outname = f'{outname}__withcen'
    if show_satellites:
        outname = f'{outname}__withsat'
    if show_ratios:
        outname = f'{outname}__ratios'
    output = os.path.join('relations', ycol, outcols, outdir, outname)
    output = save_plot(fig, output, sim, tight=False)
    txt = output.replace('plots/', 'data/').replace('.pdf', '.txt')
    os.makedirs(os.path.split(txt)[0], exist_ok=True)
    # note that this is only the mean relation, not binned by bincol
    np.savetxt(txt, np.transpose([xcenters, nsat, satrel, ncen, cenrel]),
               fmt='%.5e', header=f'{xcol} Nsat {ycol}__sat Ncen {ycol}__cen')
    return relation
