"""
Two things to do here:
    - Ratio between satellite and central HSMR as a function of time/z
    - Ratio between HSMR of things that are now satellites and were
      infalling around z and centrals at z
"""
from icecream import ic
from matplotlib import pyplot as plt, ticker
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import numpy as np
import os
from scipy.stats import (
    binned_statistic as binstat, binned_statistic_2d as binstat2d)
from time import time

from plottery.plotutils import colorscale, savefig, update_rcParams
update_rcParams()

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.hbt_tools import save_plot
from hbtpy.simulation import Simulation
from hbtpy.subhalo import HostHalos, Subhalos
from hbtpy.track import Track, HaloTracks
from hbtpy.helpers.plot_definitions import axlabel


def main():

    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation)

    to = time()
    reader = HBTReader(sim.path)
    print('Loaded reader in {0:.1f} seconds'.format(time()-to))

    # min masses are the same as in mass_relations.py
    mstarbins = np.logspace(8, 12, 13)
    ic(np.log10(mstarbins))
    # loop over this
    isnap_array = -np.arange(1, 251, 10, dtype=int)
    #isnap_array[0] = -1
    ic(isnap_array)
    chsmr, shsmr = np.zeros((2,isnap_array.size,mstarbins.size-1))
    chsmr_err, shsmr_err = np.zeros((2,isnap_array.size,2,mstarbins.size-1))
    if args.ncores > 1:
        pool = Pool(args.ncores)
        results = [
            pool.apply_async(hsmr_snap,
                             args=(args,sim,reader,isnap,mstarbins))
            for isnap in isnap_array]
        pool.close()
        pool.join()
        for out in results:
            out = out.get()
            i = (isnap_array == out[-1])
            (chsmr[i], chsmr_err[i]), (shsmr[i], shsmr_err[i]) = out[:-1]
    else:
        for i, isnap in enumerate(isnap_array):
            (chsmr[i], chsmr_err[i]), (shsmr[i], shsmr_err[i]), _ \
                = hsmr_snap(args, sim, reader, isnap, mstarbins)
    plot_hsmr_ratio(args, sim, isnap_array, mstarbins, chsmr, shsmr)
    return


def binned_mean_error_range(x, y, bins, p=(16,84)):
    #ic(bins)
    avg = binstat(x, y, 'mean', bins).statistic
    n = np.histogram(x, bins)[0]
    plo = binstat(x, y, lambda x: np.percentile(x, p[0]), bins).statistic
    phi = binstat(x, y, lambda x: np.percentile(x, p[1]), bins).statistic
    return avg - (avg-plo) / n**0.5, avg + (phi-avg) / n**0.5


def hsmr_snap(args, sim, reader, isnap, mstarbins,
              logMmin=8, logM200Mean_min=9):
    """Return the central and satellite hsmr at ``isnap``"""
    subs = Subhalos(
        reader.LoadSubhalos(isnap), sim, isnap, logMmin=logMmin,
        logM200Mean_min=logM200Mean_min, verbose_when_loading=False)
    print(f'Loaded {subs.size} isnap={isnap} subhalos!')
    c = (subs['Rank'] == 0)
    # mean and Normal error on the mean
    #asym_uncert(subs['Mstar'][c], subs['Mbound'][c], mstarbins)
    #return
    ms = subs['Mstar']
    mt = subs['Mbound']
    chsmr, shsmr = [[
        binstat(ms[mask], mt[mask], 'mean', mstarbins).statistic,
        binned_mean_error_range(ms[mask], mt[mask], mstarbins)]
        for mask in (c, ~c)]
    #ic(shsmr)
    #ic([i.shape for i in chsmr])
    return chsmr, shsmr, isnap


def plot_hsmr(args, sim, xbins, chsmr, shsmr, isnap, fig=None, axes=None):
    """Wrapper to add multiple lines to left and right panels"""
    logxbins = np.log10(xbins)
    xo = 10**((logxbins[:-1]+logxbins[1:])/2)
    if fig is None and axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(13,6), constrained_layout=True)
    elif axes is None:
        axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]


def plot_hsmr_ratio(args, sim, isnap_array, xbins, chsmr, shsmr, thin=5):
    fig, axes = plt.subplots(1, 2, figsize=(13,6), constrained_layout=True)
    # a few example curves in the left panel
    logxbins = np.log10(xbins)
    xo = 10**((logxbins[:-1]+logxbins[1:])/2)
    ic(xo.shape, shsmr.shape)
    print(f'Plotting curves for isnap={isnap_array[::thin]}')
    for j, (i, dashes) in enumerate(
            zip((0, isnap_array.size//2, isnap_array.size-1),
                ((10,0), (8,6), (4,2)))):
        color = f'{0.6-0.2*j:.1f}'
        axes[0].plot(xo, shsmr[i], color, lw=4, dashes=dashes,
                     label=f'$z={sim.redshifts[isnap_array[i]]:.1f}$')
        axes[0].plot(xo, chsmr[i], color, dashes=dashes, lw=2)
    # latest redshift
    # axes[0].plot(xo, shsmr[0], 'k-', lw=4,
    #              label=f'$z={sim.redshifts[isnap_array[0]]:.1f}$')
    # axes[0].plot(xo, chsmr[0], 'k-', lw=1)
    # # inter
    # axes[0].plot(xo, shsmr[-1], 'k--', lw=4,
    #              label=f'$z={sim.redshifts[isnap_array[-1]]:.1f}$')
    # axes[0].plot(xo, chsmr[-1], 'k--', lw=1)
    axes[0].legend()
    axes[0].set(xscale='log', yscale='log', xlabel=axlabel['Mstar'],
                ylabel=axlabel['Mbound'])
    # full ratio grid in the right panel
    # this assumes isnap_array is linearly spaced
    isnap_bins = np.append(
        isnap_array - 0.5*(isnap_array[1]-isnap_array[0]),
        isnap_array[-1] + 0.5*(isnap_array[1]-isnap_array[0]))
    ic(logxbins, logxbins.shape)
    ic(isnap_bins, isnap_bins.shape)
    ratio = shsmr / chsmr
    ic(ratio.shape)
    im = axes[1].pcolormesh(xbins, isnap_bins, ratio, cmap='viridis')
    plt.colorbar(im, ax=axes[1],
                 label=r'$m_\mathrm{sub}^{sat}/m_\mathrm{sub}^{cen}$')
    yticks = isnap_array[::isnap_array.size//5]
    yticklabels = [f'{i:.1f}' for i in sim.redshifts[yticks]]
    axes[1].set(xscale='log', xlabel=axlabel['Mstar'],
                ylabel='Redshift', yticks=yticks, yticklabels=yticklabels)
    #axes[1].yaxis.set_minor_locator(ticker.MaxNLocator(5))
    output = 'hsmr_ratio'
    output = save_plot(fig, output, sim, tight=False)
    return


def plot_hsmr_ratio_old(args, sim, xbins, chsmr, shsmr, isnap):
    axes[0].fill_between(xo, *shsmr[1], color='C0', alpha=0.2)
    axes[0].plot(xo, shsmr[0], 'C0-', lw=4, label='Satellites')
    axes[0].fill_between(xo, *chsmr[1], color='C1', alpha=0.2)
    axes[0].plot(xo, chsmr[0], 'C1--', lw=4, label='Centrals')
    axes[0].legend()
    axes[0].set(xscale='log', yscale='log', xlabel=axlabel['Mstar'],
                ylabel=axlabel['Mbound'])
    ratio = shsmr[0]/chsmr[0]
    # uncertainty in ratio
    es = np.array([shsmr[0]-shsmr[1][0], shsmr[1][1]-shsmr[0]])
    ec = np.array([chsmr[0]-chsmr[1][0], chsmr[1][1]-chsmr[0]])
    ratio_err = (es*chsmr[0] - shsmr[0]*ec) / chsmr[0]**2
    axes[1].fill_between(
    xo, ratio-ratio_err[0], ratio+ratio_err[1], color='0.8')
    axes[1].plot(xo, ratio, 'k-', lw=4)
    axes[1].set(xscale='log', xlabel=axlabel['Mstar'],
                ylabel=r'$m_\mathrm{sub}^{sat}/m_\mathrm{sub}^{cen}$')
    output = 'hsmr_curves'
    output = save_plot(fig, output, sim, tight=False)
    return


main()
