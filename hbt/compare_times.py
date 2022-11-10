from astropy.io import ascii, fits
from astropy.units import Quantity
import cmasher as cmr
from glob import glob
from icecream import ic
from itertools import count
from matplotlib import (
    cm, colors as mplcolors, pyplot as plt, ticker, rcParams)
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from mpl_toolkits.axes_grid1 import ImageGrid
import multiprocessing as mp
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic_2d, pearsonr
import sys
from time import sleep, time

from plottery.plotutils import colorscale, savefig, update_rcParams
update_rcParams()
rcParams['text.latex.preamble'] += r',\usepackage{color}'

from HBTReader import HBTReader

# local
from hbtpy import hbt_tools
from hbtpy.helpers.plot_definitions import axlabel, binlabel
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos#, Track
from hbtpy.track import Track

adjust_kwargs = dict(
    left=0.10, right=0.95, bottom=0.05, top=0.98, wspace=0.3, hspace=0.1)


def main():
    print('Running...')
    args = (
        ('--demographics', {'action': 'store_true'},),
        ('--investigate', {'action': 'store_true'})
        )
    args = hbt_tools.parse_args(args=args)
    sim = Simulation(args.simulation)

    to = time()
    reader = HBTReader(sim.path)
    print(f'Loaded reader in {time()-to:.1f} seconds')
    to = time()
    subs = Subhalos(
        reader.LoadSubhalos(-1), sim, -1, as_dataframe=True, logMmin=9,
        logM200Mean_min=9)
    #subs.sort(order='Mbound')
    print(f'Loaded subhalos in {(time()-to)/60:.2f} minutes')

    print('In total there are {0} central and {1} satellite subhalos'.format(
        subs.centrals.size, subs.satellites.size))

    centrals = Subhalos(
        subs.centrals, sim, -1, load_distances=False, load_velocities=False,
        load_history=False, logMstar_min=9)
    satellites = Subhalos(
        subs.satellites, sim, -1, load_distances=False, load_velocities=False,
        load_history=False, logM200Mean_min=13, logMstar_min=9)
    print(np.sort(satellites.colnames))

    plot_times(
        satellites, c_lower='corr', c_upper='Mstar',
        vmin_upper=9.3, vmax_upper=10.7)
    return


def plot_times(satellites, c_lower='corr', c_upper='Mstar', use_lookback=True,
               cmap_lower='cmr.amber_r', cmap_lower_rng=(0,0.8),
               cmap_upper='cmr.toxic_r', cmap_upper_rng=(0,1),
               vmin_lower=None, vmax_lower=None, stat_lower=np.nanmean,
               vmin_upper=None, vmax_upper=None, stat_upper=np.nanmean):
    #cmap_lower = define_cmap(c_lower, cmap_lower, cmap_lower_rng)
    events = ('cent', 'sat', 'first_infall', 'last_infall',
              'max_Mbound', 'max_Mstar')
    axlabels = ['cent', 'sat', 'infall', 'acc',
                '$m_\mathrm{sub}^\mathrm{max}$',
                '$m_\mathrm{\u2605}^\mathrm{max}$']
    axlabels = [f'{i} (Gyr)' for i in axlabels]
    nc = len(events)
    fig, axes = plt.subplots(
        nc, nc, figsize=(2*nc,2.3*nc), constrained_layout=True)
    tx = np.arange(0, 13.6, 0.5)
    extent = (tx[0], tx[-1], tx[0], tx[-1])
    xlim = extent[:2]
    # to convert lookback times into Universe ages
    tmax = 13.7
    iname = 0
    for i, ev_i in enumerate(events):
        xcol = f'history:{ev_i}:time'
        x = satellites[xcol] if use_lookback else tmax - satellites[xcol]
        for j, ev_j in enumerate(events):
            ax = axes[i,j]
            if j == 0:
                ax.set_ylabel(axlabels[i], fontsize=18)
            else:
                ax.set(yticklabels=[])
            if i == nc - 1:
                ax.set_xlabel(axlabels[j], fontsize=18)
            else:
                ax.set(xticklabels=[])
            compare_times_ticks(ax, diagonal=(i == j))
            ycol = f'history:{ev_j}:time'
            y = satellites[ycol] if use_lookback else tmax - satellites[ycol]
            if j < i:
                if c_lower is None:
                    ax.axis('off')
                else:
                    im_lower = plot_times_2d(
                        satellites, x, y, tx, ax, i, j, iname, extent, c_lower,
                        cmap=cmap_lower, cmap_rng=cmap_lower_rng,
                        vmin=vmin_lower, vmax=vmax_lower, stat=stat_lower)
            # diagonal
            elif j == i:
                plot_times_hist(x, tx, ax, i, iname, xlim)
                if i < nc - 1:
                    ax.set(xticks=[])
                else:
                    xcol = f'history:{events[i]}:time'
                    ax.set_xlabel(axlabels[i], fontsize=18)
            # upper triangle
            elif j > i:
                if c_upper is None:
                    ax.axis('off')
                else:
                    im_upper = plot_times_2d(
                        satellites, x, y, tx, ax, i, j, iname, extent, c_upper,
                        cmap=cmap_upper, cmap_rng=cmap_upper_rng,
                        vmin=vmin_upper, vmax=vmax_upper, stat=stat_upper)
            ax.set(xlim=xlim)
            if j != i:
                ax.plot(tx, tx, 'k--', lw=1)
                ax.set(ylim=xlim)
                ax.tick_params(which='both', length=3)
            iname += 1
    # for the colorbars we're assuming that correlations would
    # only happen in the lower-left, for now
    # lower-left off-diagonal colorbar
    show_colorbar(
        axes, im_lower, c_lower, cmap_lower, stat_lower,
        orientation='horizontal', location='bottom')
    show_colorbar(
        axes, im_upper, c_upper, cmap_upper, stat_upper,
        orientation='horizontal', location='top')
    # save!
    output = f'correlations/compare_times_{c_lower}_{c_upper}'
    hbt_tools.save_plot(
        fig, output, satellites.sim, tight=False, h_pad=0.2)
    return


def compare_times_ticks(ax, diagonal=False):
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_major_locator(ticker.FixedLocator([5,10]))
    if diagonal:
        ax.yaxis.set_major_locator(ticker.NullLocator())
    else:
        ax.yaxis.set_major_locator(ticker.FixedLocator([5,10]))
    return


def define_cmap(c, cmap, cmap_rng):
    if c == 'corr':
        return cmr.get_sub_cmap(cmap_lower, 0, 0.8)


def plot_times_2d(satellites, x, y, bins, ax, i, j, axname, extent,
                  c, cmap, cmap_rng=(0,1),
                  stat=np.nanmean, vmin=None, vmax=None, annotate_r=True):
    # these are for correlations and for annotations
    r, pr = pearsonr(x, y)
    h2d = np.histogram2d(x, y, bins=bins)[0]
    if c == 'corr':
        cmap = cmr.get_sub_cmap(cmap, *cmap_rng)
        color = cmr.take_cmap_colors(
            cmap, N=1, cmap_range=(r,r))[0]
        cmap_ij = mplcolors.LinearSegmentedColormap.from_list(
            'cmap_ij', [[1, 1, 1], color])
        im = ax.imshow(
            h2d, extent=extent, cmap=cmap_ij,
            origin='lower', aspect='auto', vmin=0, vmax=0.03*h2d.sum())
    else:
        cmap = cmr.get_sub_cmap(cmap, *cmap_rng)
        m2d = binned_statistic_2d(
            x, y, satellites[c], stat, bins=bins)[0]
        im = ax.imshow(
            np.log10(m2d.T), origin='lower', aspect='auto', vmin=vmin,
            vmax=vmax, cmap=cmap, extent=extent)
    # annotate correlation coefficient
    if annotate_r:
        if np.triu(h2d.T, 2).sum()/h2d.sum() < 0.25:
            label = f'{r:.2f}\n({axname})'
            xy = (0.05, 0.95)
            ha, va = 'left', 'top'
        else:
            label = f'({axname})\n{r:.2f}'
            xy = (0.95, 0.05)
            ha, va = 'right', 'bottom'
        ax.annotate(
            label, xy=xy, xycoords='axes fraction',
            ha=ha, va=va, fontsize=14)
    return im


def plot_times_hist(x, tx, ax, i, iname, xlim):
    ax.annotate(
        f'({iname})', xy=(0.05,0.95), xycoords='axes fraction',
        ha='left', va='top', fontsize=14)
    ax.hist(x, tx, histtype='stepfilled', color='C9')
    ax.axvline(np.median(x), color='0.2')
    ax.tick_params(which='both', length=5)
    ax.set(yticks=[], xlim=xlim)
    return


def show_colorbar(axes, im, c, cmap, stat, orientation='vertical',
                  location='left', logstat=True):
    if c == 'corr':
        cbar = cm.ScalarMappable(
            norm=mplcolors.Normalize(vmin=0.2, vmax=1), cmap=cmap)
        cbar = plt.colorbar(
            cbar, ax=axes, location=location, fraction=0.1, aspect=30,
            label='Correlation')
    else:
        if isinstance(stat, str):
            statlabel = stat
        elif stat in (np.mean, np.nanmean):
            statlabel = 'mean'
        elif stat in (np.median, np.nanmedian):
            statlabel = 'median'
        if logstat:
            statlabel = f'log {statlabel}'
        plt.colorbar(
            im, ax=axes, location=location, fraction=0.1, aspect=30,
            label=f'{statlabel}(${binlabel[c]}$)')
    return


main()