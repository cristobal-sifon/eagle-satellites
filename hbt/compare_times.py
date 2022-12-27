import cmasher as cmr
from icecream import ic
from matplotlib import (
    cm, colors as mplcolors, pyplot as plt, ticker, rcParams)
import multiprocessing as mp
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic, binned_statistic_2d, pearsonr
from time import time

from plottery.plotutils import savefig, update_rcParams
update_rcParams()
rcParams['text.latex.preamble'] += r',\usepackage{color}'

from HBTReader import HBTReader

# local
from hbtpy import hbt_tools
from hbtpy.helpers.plot_definitions import binlabel
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos

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

    review_subsamples(args, satellites)
    return

    if args.ncores > 1:
        pool = mp.Pool()
        run = pool.apply_async
    else:
        run = lambda f: f
    run(plot_times(
        satellites, c_lower='corr', c_upper='Mstar',
        cmap_lower='cmr.ember_r', cmap_lower_rng=(0.35, 1),
        cmap_upper='cmr.toxic_r',
        vmin_upper=9.3, vmax_upper=11))
    run(plot_times(
        satellites, c_lower='Mbound', c_upper='history:first_infall:Mbound',
        cmap_lower='cmr.ember_r', cmap_upper='cmr.cosmic_r'))
    run(plot_times(
        satellites, c_lower='history:first_infall:Mbound',
        c_upper='history:first_infall:Mbound/history:first_infall:Mstar',
        cmap_lower='cmr.ember_r', cmap_upper='cmr.cosmic_r',
        vmin_lower=10.5, vmax_lower=13, cmap_lower_rng=(0,0.8),
        cmap_upper_rng=(0,0.8)))
    run(plot_times(
        satellites, c_lower='history:first_infall:Mstar',
        c_upper='M200Mean'))
    run(plot_times(
        satellites, c_lower='history:sat:Mstar', c_upper='Mstar',
        cmap_lower='cmr.toxic_r', cmap_upper='cmr.toxic_r',
        vmin_lower=9.5, vmax_lower=11.5, vmin_upper=9.3, vmax_upper=11))
    if args.ncores > 1:
        pool.close()
        pool.join()
    return


def review_subsamples(args, s):
    plot_path = os.path.join(s.sim.plot_path, 'compare_times')
    tlb = s.sim.t_lookback.value
    print('\n*** Special subsamples ***\n')
    nsat = s.size
    ic(nsat)
    review_tmstar(args, s, tlb, nsat, plot_path)
    review_tsat(args, s, tlb, nsat, plot_path)
    review_tacc(args, s, tlb, nsat, plot_path)
    return


def review_tmstar(args, s, tlb, nsat, plot_path):
    print('** max mstar **')
    tmstar = s['history:max_Mstar:time']
    tlate = tlb[-2]
    ic(s.sim.t_lookback[-20:])
    hmstar, hmstar_bins = np.histogram(tmstar, tlb[-30:][::-1])
    ic(hmstar_bins)
    ic(hmstar)
    ic(hmstar[0]/nsat)
    late_mstar = (tmstar < tlate)
    ic(late_mstar.sum())
    diffbins = np.arange(-13.5, 13.6, 0.5)
    ic(diffbins)
    for event in ('cent', 'sat', 'first_infall', 'last_infall',
                  'max_Mbound', 'max_Mgas'):
        h = f'history:{event}'
        ti = s[f'{h}:time']
        ic(h, np.median(ti[late_mstar]),
           np.median(ti[~late_mstar]))
        ic((ti > tmstar).sum(), (ti > tmstar).sum()/nsat)
        ic(np.median(tmstar[late_mstar] - ti[late_mstar]))
        ic(np.median(tmstar[~late_mstar] - ti[~late_mstar]))
        ic(binned_statistic(
            tmstar, ti-tmstar, np.nanmedian, diffbins)[0])
        ic(binned_statistic(
            tmstar[late_mstar], (ti-tmstar)[late_mstar],
            np.nanmedian, diffbins)[0])
        ic(binned_statistic(
            tmstar[~late_mstar], (ti-tmstar)[~late_mstar],
            np.nanmedian, diffbins)[0])
    for m in ('Mbound', 'Mstar', 'M200Mean'):
        ic(m, np.log10(np.median(s[m][late_mstar])),
           np.log10(np.median(s[m][~late_mstar])))
    fit, cov = curve_fit(
        lambda x, a, b: a+b*x, s['history:max_Mbound:time'][~late_mstar],
        s['history:max_Mstar:time'][~late_mstar], p0=(5.1,0.3))
    ic(fit, np.diag(cov)**0.5)
    return


def review_tsat(args, s, tlb, nsat, plot_path):
    print('** t_sat **')
    tsat = s['history:sat:time']
    ic(tlb[:20])
    htsat, htsat_bins = np.histogram(tsat, tlb[::-1])
    # fig, ax = plt.subplots(constrained_layout=True)
    # ax.plot(htsat_bins[1:], np.cumsum(htsat)/nsat)
    # ax.set(xlabel='$t_\mathrm{sat}^\mathrm{lookback}$ (Gyr)',
    #        ylabel='$N(<t_\mathrm{sat}^\mathrm{lookback})$')
    # output = os.path.join(plot_path, 'tsat_cdf.png')
    # savefig(output, fig=fig, tight=False)
    # ic(htsat_bins)
    # ic(htsat)
    #ic(np.cumsum(htsat[::-1])/nsat)
    early_tsat = (tsat > 12)
    nearly = early_tsat.sum()
    ic(nearly, nearly/nsat)
    # fig, ax = plt.subplots(figsize=(5,4))
    # ax.hist(s['history'])
    for event in ('cent', 'sat', 'first_infall', 'last_infall',
                  'max_Mbound', 'max_Mgas', 'max_Mstar'):
        h = f'history:{event}'
        ic(h, np.median(s[f'{h}:time'][early_tsat]),
           np.median(s[f'{h}:time'][~early_tsat]))
    for m in ('Mbound', 'Mstar', 'M200Mean'):
        ic(m, np.log10(np.median(s[m][early_tsat])),
           np.log10(np.median(s[m][~early_tsat])))


def review_tacc(args, s, tlb, nsat, plot_path):
    print('** t_acc **')
    tacc = s['history:last_infall:time']
    htacc, htacc_bins = np.histogram(tacc, tlb[-30:][::-1])
    ic(htacc_bins)
    ic(htacc)
    j = np.argmax(htacc)
    j1 = np.s_[j+1:j+2]
    j2 = np.s_[j+2:j+3]
    spike1 = (tacc >= htacc_bins[j1][0]) & (tacc <= htacc_bins[j1][-1])
    ic(htacc_bins[j1], htacc[j1])
    ic(spike1.sum(), spike1.sum()/nsat)
    ic(np.unique(s['HostHaloId'][spike1], return_counts=True))
    ic((s['HostHaloId'] == 1).sum())
    spike2 = (tacc >= htacc_bins[j2][0]) & (tacc <= htacc_bins[j2][-1])
    ic(htacc_bins[j2], htacc[j2])
    ic(spike2.sum(), spike2.sum()/nsat)
    ic(np.unique(s['HostHaloId'][spike2], return_counts=True))
    ic((s['HostHaloId'] == 5).sum())
    return


def plot_times(satellites, c_lower='corr', c_upper='Mstar', use_lookback=True,
               cmap_lower='cmr.ember_r', cmap_lower_rng=(0,1),
               cmap_upper='cmr.cosmic_r', cmap_upper_rng=(0,1), vmin_lower=None,
               vmax_lower=None, stat_lower=np.nanmean, vmin_upper=None,
               vmax_upper=None, stat_upper=np.nanmean):
    #cmap_lower = define_cmap(c_lower, cmap_lower, cmap_lower_rng)
    events = ('cent', 'sat', 'first_infall', 'last_infall', 'max_Mbound',
              'max_Mstar', 'max_Mgas')
    nc = len(events)
    fig, axes = plt.subplots(
        nc, nc, figsize=(2*nc,2.4*nc), constrained_layout=True)
    tx = np.arange(0, 13.6, 0.5)
    extent = (tx[0], tx[-1], tx[0], tx[-1])
    xlim = extent[:2]
    # to convert lookback times into Universe ages
    tmax = 13.7
    iname = 1
    for i, ev_i in enumerate(events):
        xcol = f'history:{ev_i}:time'
        x = satellites[xcol] if use_lookback else tmax - satellites[xcol]
        m = ['Mbound', 'Mstar', 'Mgas', 'M200Mean']
        m += [f'history:{ev}:{mi}' for mi in m for ev in ('birth',)+events]
        ic('---')
        ic(xcol)
        for mi in m:
            r = pearsonr(x, satellites[mi])[0]
            #if abs(r) > 0.5:
            ic(mi, r)
        ic('---')
        for j, ev_j in enumerate(events):
            ax = axes[i,j]
            format_ax(ax, i, j, xlim, nc)
            ycol = f'history:{ev_j}:time'
            y = satellites[ycol] if use_lookback else tmax - satellites[ycol]
            # diagonal
            if j == i:
                plot_times_hist(x, tx, ax, iname, xlim)
            # lower triangle
            elif j < i:
                if c_lower is None:
                    ax.axis('off')
                else:
                    im_lower = plot_times_2d(
                        satellites, x, y, tx, ax, i, j, iname, extent, c_lower,
                        cmap=cmap_lower, cmap_rng=cmap_lower_rng,
                        vmin=vmin_lower, vmax=vmax_lower, stat=stat_lower)
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
    # colorbars
    show_colorbar(
        axes, im_lower, c_lower, cmap_lower, cmap_lower_rng, stat_lower,
        location='bottom')
    show_colorbar(
        axes, im_upper, c_upper, cmap_upper, cmap_upper_rng, stat_upper,
        location='top')
    # save!
    c_lower = hbt_tools.format_colname(c_lower)
    c_upper = hbt_tools.format_colname(c_upper)
    output = f'correlations/comparetimes__{c_lower}__{c_upper}'
    hbt_tools.save_plot(
        fig, output, satellites.sim, tight=False, h_pad=0.2)
    return


def format_ax(ax, i, j, xlim, ncols, fs=18, labelpad=5):
    axlabels = ['cent', 'sat', 'infall', 'acc',
                '$m_\mathrm{sub}^\mathrm{max}$',
                '$m_\mathrm{\u2605}^\mathrm{max}$',
                '$m_\mathrm{gas}^\mathrm{max}$']
    axlabels = [f'{i} (Gyr)' for i in axlabels]
    kwargs = {'fontsize': fs, 'labelpad': labelpad}
    # diagonal
    if i == j and i < ncols - 1:
        ax.set(xticks=[])
    # else:
    #     xcol = f'history:{events[i]}:time'
    #     ax.set_xlabel(axlabels[i], **kwargs)
    # left
    if j == 0 and i > 0:
        ax.set_ylabel(axlabels[i], **kwargs)
    else:
        ax.set(yticklabels=[])
    # right
    if j == ncols - 1 and i < ncols - 1:
        rax = ax.twinx()
        rax.plot([], [])
        rax.set(ylim=xlim)
        rax.set_ylabel(axlabels[i], **kwargs)
        format_ticks(rax)
    # bottom
    if i == ncols - 1 and j < ncols - 1:
        ax.set_xlabel(axlabels[j], **kwargs)
    else:
        ax.set(xticklabels=[])
    # top
    if i == 0 and j > 0:
        tax = ax.twiny()
        tax.plot([], [])
        tax.set(xlim=xlim)
        kwargs['labelpad'] += 2
        tax.set_xlabel(axlabels[j], **kwargs)
        format_ticks(tax)
    format_ticks(ax, (i == j))
    return


def format_ticks(ax, diagonal=False):
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_major_locator(ticker.FixedLocator([5,10]))
    if diagonal:
        ax.yaxis.set_major_locator(ticker.NullLocator())
    else:
        ax.yaxis.set_major_locator(ticker.FixedLocator([5,10]))
    return


def plot_times_2d(satellites, x, y, bins, ax, i, j, axname, extent,
                  c, cmap, cmap_rng=(0,1),
                  stat=np.nanmean, vmin=None, vmax=None, annotate_r=True):
    is_lower = j < i
    # these are for correlations and for annotations
    r, pr = pearsonr(x, y)
    h2d = np.histogram2d(x, y, bins=bins)[0]
    ntot = x.size
    if c == 'corr':
        cmap = cmr.get_sub_cmap(cmap, *cmap_rng)
        color = cmr.take_cmap_colors(
            cmap, N=1, cmap_range=(r,r))[0]
        cmap_ij = mplcolors.LinearSegmentedColormap.from_list(
            'cmap_ij', [[1, 1, 1], color])
        # making vmin slightly <0 so that the colormaps don't include white
        # but those equal to zero should be nan so empty space is still white
        #h2d[h2d == 0] = np.nan
        im = ax.imshow(
            h2d, extent=extent, cmap=cmap_ij,
            origin='lower', aspect='auto', vmin=0, vmax=0.025*ntot)
        h2d[np.isnan(h2d)] = 0
    else:
        cmap = cmr.get_sub_cmap(cmap, *cmap_rng)
        m2d = binned_statistic_2d(
            x, y, satellites[c], stat, bins=bins)[0]
        #if not is_lower:
            #m2d = m2d.T
        im = ax.imshow(
            np.log10(m2d), origin='lower', aspect='auto', vmin=vmin,
            vmax=vmax, cmap=cmap, extent=extent)
    # annotate correlation coefficient
    if annotate_r:
        # annot_top = (f'{r:.2f}\n({axname})', (0.05, 0.95), 'left', 'top')
        # annot_bottom = (f'({axname})\n{r:.2f}', (0.95, 0.05), 'right', 'bottom')
        annot_top = (f'{r:.2f}', (0.03, 0.97), 'left', 'top')
        annot_bottom = (f'{r:.2f}', (0.97, 0.03), 'right', 'bottom')
        ic(axname, is_lower, r, np.triu(h2d, 2).sum()/h2d.sum())
        if np.triu(h2d.T, 2).sum()/h2d.sum() < 0.2:
            annot = annot_top# if is_lower else annot_bottom
            #annot = annot_top
        else:
            annot = annot_bottom# if is_lower else annot_top
            #annot = annot_top
        label, xy, ha, va = annot
        ax.annotate(
            label, xy=xy, xycoords='axes fraction',
            ha=ha, va=va, fontsize=14)
    return im


def plot_times_hist(x, tx, ax, axname, xlim):
    # ax.annotate(
    #     f'({iname})', xy=(0.05,0.95), xycoords='axes fraction',
    #     ha='left', va='top', fontsize=14)
    h = ax.hist(x, tx, histtype='stepfilled', color='C9')[0]
    ic(axname, tx, h, h/h.sum())
    ax.axvline(np.median(x), color='0.2')
    ax.tick_params(which='both', length=5)
    ax.set(yticks=[], xlim=xlim)
    return


def show_colorbar(axes, im, c, cmap, cmap_rng, stat, location='left',
                  logstat=True):
    assert location in ('left', 'right', 'bottom', 'top')
    orientation = 'vertical' if location in ('left', 'right') else 'horizontal'
    if c == 'corr':
        cmap = cmr.get_sub_cmap(cmap, *cmap_rng)
        cbar = cm.ScalarMappable(
            norm=mplcolors.Normalize(vmin=0.2, vmax=0.9), cmap=cmap)
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
        if '/' in c:
            c = c.split('/')
            label = f'{binlabel[c[0]]}/{binlabel[c[1]]}'
        else:
            label = binlabel[c]
        cbar = plt.colorbar(
            im, ax=axes, location=location, fraction=0.1, aspect=30,
            label=f'{statlabel}(${label}$)')
        #cbar.
    return


main()
